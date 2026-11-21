import time
import struct
import zlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SearchResult:
    id: int
    title: str
    category: str
    price: float
    distance: float


@dataclass
class SearchStats:
    latency_ms: float
    distance_computations: int
    candidates_evaluated: int


class VectorSimilarityStore:
    def __init__(self, dim: int = 32, nlist: int = 16, metric: str = "cosine"):
        self.dim = dim
        self.nlist = nlist
        self.metric = metric
        self.vectors: Optional[np.ndarray] = None
        self.titles: List[str] = []
        self.categories: np.ndarray = np.array([])
        self.prices: np.ndarray = np.array([])
        self.centroids: Optional[np.ndarray] = None
        self.inv_lists: List[np.ndarray] = []

    def generate_dataset(self, n: int = 10000, seed: int = 42):
        rng = np.random.default_rng(seed)
        num_centers = min(16, n)
        centers = rng.normal(0, 3.0, size=(num_centers, self.dim)).astype(np.float32)
        cluster_ids = np.arange(n) % num_centers
        noise = rng.normal(0, 0.65, size=(n, self.dim)).astype(np.float32)
        self.vectors = centers[cluster_ids] + noise

        cats = ["Electronics", "Books", "Audio", "Fitness", "Home"]
        self.categories = np.array([cats[i % len(cats)] for i in range(n)])
        self.prices = np.round(rng.uniform(9.99, 499.99, size=n), 2)
        self.titles = [f"{self.categories[i]} Product #{i}" for i in range(n)]
        self._build_ivf(seed=seed)

    def _compute_distances(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 0:
            return np.array([], dtype=np.float32)
        if self.metric == "l2":
            diff = matrix - query
            return np.sum(diff * diff, axis=1)
        else:
            q_norm = np.linalg.norm(query) + 1e-12
            m_norms = np.linalg.norm(matrix, axis=1) + 1e-12
            dots = matrix @ query
            return 1.0 - (dots / (m_norms * q_norm))

    def _build_ivf(self, max_iters: int = 10, seed: int = 42):
        n = len(self.vectors)
        k = min(self.nlist, n)
        self.nlist = k
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n, size=k, replace=False)
        centroids = self.vectors[init_idx].copy()

        assignments = np.zeros(n, dtype=np.int32)
        for _ in range(max_iters):
            # Assign vectors to nearest centroid (L2)
            dists = np.sum((self.vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_assign = np.argmin(dists, axis=1)
            if np.array_equal(assignments, new_assign):
                break
            assignments = new_assign
            for c in range(k):
                mask = assignments == c
                if np.any(mask):
                    centroids[c] = self.vectors[mask].mean(axis=0)

        self.centroids = centroids
        self.inv_lists = [np.where(assignments == c)[0] for c in range(k)]

    def _filter_mask(self, category: str, max_price: float) -> np.ndarray:
        mask = np.ones(len(self.vectors), dtype=bool)
        if category and category != "ALL":
            mask &= (self.categories == category)
        if max_price >= 0:
            mask &= (self.prices <= max_price)
        return mask

    def search_flat(self, query: np.ndarray, k: int = 8, category: str = "ALL", max_price: float = -1.0) -> Tuple[List[SearchResult], SearchStats]:
        t0 = time.perf_counter()
        mask = self._filter_mask(category, max_price)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return [], SearchStats((time.perf_counter() - t0) * 1000, 0, 0)

        dists = self._compute_distances(query, self.vectors[valid_idx])
        top_local = np.argsort(dists)[:k]
        chosen = valid_idx[top_local]

        results = [
            SearchResult(int(i), self.titles[i], str(self.categories[i]), float(self.prices[i]), float(dists[loc]))
            for i, loc in zip(chosen, top_local)
        ]
        lat = (time.perf_counter() - t0) * 1000.0
        return results, SearchStats(lat, len(valid_idx), len(valid_idx))

    def search_ivf(self, query: np.ndarray, k: int = 8, nprobe: int = 4, filter_mode: str = "pre",
                   category: str = "ALL", max_price: float = -1.0, overfetch: int = 4) -> Tuple[List[SearchResult], SearchStats]:
        t0 = time.perf_counter()
        probes = min(max(1, nprobe), self.nlist)
        c_dists = np.sum((self.centroids - query) ** 2, axis=1)
        nearest_clusters = np.argsort(c_dists)[:probes]

        cand_idx = np.concatenate([self.inv_lists[c] for c in nearest_clusters]) if probes > 0 else np.array([], dtype=int)
        dist_comps = self.nlist

        if len(cand_idx) == 0:
            return [], SearchStats((time.perf_counter() - t0) * 1000, dist_comps, 0)

        if filter_mode == "pre":
            if category != "ALL":
                cand_idx = cand_idx[self.categories[cand_idx] == category]
            if max_price >= 0:
                cand_idx = cand_idx[self.prices[cand_idx] <= max_price]
            if len(cand_idx) == 0:
                return [], SearchStats((time.perf_counter() - t0) * 1000, dist_comps, 0)
            dists = self._compute_distances(query, self.vectors[cand_idx])
            dist_comps += len(cand_idx)
            top_local = np.argsort(dists)[:k]
            chosen = cand_idx[top_local]
            chosen_dists = dists[top_local]
        else:
            # Post-filter: compute distances first, overfetch k * overfetch, then filter
            dists = self._compute_distances(query, self.vectors[cand_idx])
            dist_comps += len(cand_idx)
            top_over = np.argsort(dists)[: k * overfetch]
            over_idx = cand_idx[top_over]
            over_dists = dists[top_over]

            keep = np.ones(len(over_idx), dtype=bool)
            if category != "ALL":
                keep &= (self.categories[over_idx] == category)
            if max_price >= 0:
                keep &= (self.prices[over_idx] <= max_price)
            chosen = over_idx[keep][:k]
            chosen_dists = over_dists[keep][:k]

        results = [
            SearchResult(int(i), self.titles[i], str(self.categories[i]), float(self.prices[i]), float(d))
            for i, d in zip(chosen, chosen_dists)
        ]
        lat = (time.perf_counter() - t0) * 1000.0
        return results, SearchStats(lat, dist_comps, len(cand_idx))

    def verify_crc32_persistence(self) -> Tuple[bool, int, str]:
        raw = self.vectors.tobytes()
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        header = struct.pack("<4sIII", b"P5VS", len(self.vectors), self.dim, crc)
        blob = header + raw
        magic, n, d, stored_crc = struct.unpack("<4sIII", blob[:16])
        recomputed = zlib.crc32(blob[16:]) & 0xFFFFFFFF
        return (magic == b"P5VS" and stored_crc == recomputed), len(blob), hex(crc)
