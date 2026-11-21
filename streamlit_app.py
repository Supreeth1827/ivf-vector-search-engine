import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from engine import VectorSimilarityStore

st.set_page_config(page_title="P5 Vector Search Engine", layout="wide")

st.title("P5: Similarity Search & Recommendation Engine")
st.caption("AI-Driven Software Engineering & Project Management (24CS3502) | IVF-Flat vs. Exact Flat | Hybrid Metadata Filtering")


@st.cache_resource
def load_store(n: int, dim: int, nlist: int, metric: str):
    store = VectorSimilarityStore(dim=dim, nlist=nlist, metric=metric)
    store.generate_dataset(n=n, seed=42)
    return store


tab1, tab2, tab3 = st.tabs(["Live Similarity Search", "Benchmark & 3 Tradeoff Graphs", "Storage & Durability Check"])

with tab1:
    with st.sidebar:
        st.header("Engine & Query Settings")
        n_items = st.selectbox("Dataset Size (Vectors)", [5000, 15000, 30000], index=1)
        metric = st.selectbox("Distance Metric", ["cosine", "l2"], index=0)
        query_id = st.number_input("Query Seed Item ID", min_value=0, max_value=n_items - 1, value=42)
        index_mode = st.radio("Index Strategy", ["IVF-Flat (Approximate ANN)", "Exact Flat Scan (Ground Truth)"])
        nprobe = st.slider("nprobe (Clusters Searched)", min_value=1, max_value=16, value=4)
        filter_mode = st.radio("Metadata Filter Mode", ["pre", "post"], format_func=lambda x: "Pre-Filter" if x == "pre" else "Post-Filter")
        category = st.selectbox("Category Filter", ["ALL", "Electronics", "Books", "Audio", "Fitness", "Home"])
        use_price = st.checkbox("Enable Max Price Filter", value=False)
        max_price = st.slider("Max Price ($)", 20.0, 500.0, 250.0) if use_price else -1.0
        top_k = st.slider("Top-K Results", 1, 20, 8)

    store = load_store(n_items, 32, 16, metric)
    query_vec = store.vectors[query_id]

    gt_results, gt_stats = store.search_flat(query_vec, k=top_k, category=category, max_price=max_price)
    if index_mode.startswith("Exact"):
        results, stats = gt_results, gt_stats
    else:
        results, stats = store.search_ivf(query_vec, k=top_k, nprobe=nprobe, filter_mode=filter_mode, category=category, max_price=max_price)

    gt_ids = {r.id for r in gt_results}
    hits = sum(1 for r in results if r.id in gt_ids)
    recall = (hits / len(gt_ids)) if len(gt_ids) > 0 else 1.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Query Latency", f"{stats.latency_ms:.3f} ms")
    c2.metric("Recall@K vs Exact", f"{recall * 100:.1f}%")
    c3.metric("Distance Computations", f"{stats.distance_computations:,}")
    c4.metric("Speedup vs Exact", f"{(gt_stats.latency_ms / max(stats.latency_ms, 1e-4)):.2f}x")

    st.subheader(f"Top-{top_k} Nearest Recommendations (Query Item: {store.titles[query_id]})")
    if results:
        df_res = pd.DataFrame([
            {"Rank": idx + 1, "Item ID": r.id, "Title": r.title, "Category": r.category, "Price ($)": r.price, "Distance": round(r.distance, 5)}
            for idx, r in enumerate(results)
        ])
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.warning("No items matched your filter criteria.")

with tab2:
    st.subheader("Automated Benchmark Harness (Generates the 3 Required Report Graphs)")
    if st.button("Run Multi-Trial Benchmark Now"):
        bench_store = load_store(15000, 32, 16, "l2")
        rng = np.random.default_rng(101)
        q_ids = rng.choice(15000, size=30, replace=False)

        # Graph 1: Recall@10 vs Latency across nprobe
        gts = [bench_store.search_flat(bench_store.vectors[q], k=10)[0] for q in q_ids]
        nprobes = [1, 2, 4, 8, 12, 16]
        lats, recs = [], []
        for np_val in nprobes:
            l_list, r_list = [], []
            for q, gt in zip(q_ids, gts):
                res, st_ivf = bench_store.search_ivf(bench_store.vectors[q], k=10, nprobe=np_val)
                l_list.append(st_ivf.latency_ms)
                gt_set = {x.id for x in gt}
                r_list.append(sum(1 for x in res if x.id in gt_set) / 10.0)
            lats.append(np.mean(l_list))
            recs.append(np.mean(r_list))

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(lats, recs, "-o", color="#0284c7", linewidth=2)
        for np_v, x_v, y_v in zip(nprobes, lats, recs):
            ax1.annotate(f"nprobe={np_v}", (x_v, y_v), xytext=(4, -10), textcoords="offset points", fontsize=8)
        ax1.set_title("Graph 1: Recall@10 vs. Query Latency Tradeoff")
        ax1.set_xlabel("Mean Query Latency (ms)")
        ax1.set_ylabel("Mean Recall@10")
        ax1.grid(True)

        # Graph 2: Pre-filter vs Post-filter across selectivity
        price_limits = [("10% Pass", 60.0), ("30% Pass", 160.0), ("60% Pass", 305.0), ("100% Pass", -1.0)]
        pre_rec, post_rec = [], []
        for _, p_lim in price_limits:
            pr_l, po_l = [], []
            for q in q_ids[:20]:
                gt_f, _ = bench_store.search_flat(bench_store.vectors[q], k=10, max_price=p_lim)
                if not gt_f:
                    continue
                gt_s = {x.id for x in gt_f}
                r_pre, _ = bench_store.search_ivf(bench_store.vectors[q], k=10, nprobe=4, filter_mode="pre", max_price=p_lim)
                r_post, _ = bench_store.search_ivf(bench_store.vectors[q], k=10, nprobe=4, filter_mode="post", max_price=p_lim)
                pr_l.append(sum(1 for x in r_pre if x.id in gt_s) / len(gt_s))
                po_l.append(sum(1 for x in r_post if x.id in gt_s) / len(gt_s))
            pre_rec.append(np.mean(pr_l))
            post_rec.append(np.mean(po_l))

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x_idx = np.arange(len(price_limits))
        ax2.bar(x_idx - 0.18, pre_rec, width=0.35, label="Pre-Filter", color="#0284c7")
        ax2.bar(x_idx + 0.18, post_rec, width=0.35, label="Post-Filter", color="#f59e0b")
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels([x[0] for x in price_limits])
        ax2.set_ylabel("Recall@10")
        ax2.set_title("Graph 2: Pre-Filter vs. Post-Filter Recall Across Selectivity")
        ax2.legend()
        ax2.grid(True, axis="y")

        # Graph 3: Scaling with Dataset Size N
        sizes = [2000, 8000, 15000, 30000]
        flat_times, ivf_times = [], []
        for sz in sizes:
            s_tmp = load_store(sz, 32, 16, "l2")
            f_t, i_t = [], []
            for q in range(15):
                _, st_f = s_tmp.search_flat(s_tmp.vectors[q], k=10)
                _, st_i = s_tmp.search_ivf(s_tmp.vectors[q], k=10, nprobe=4)
                f_t.append(st_f.latency_ms)
                i_t.append(st_i.latency_ms)
            flat_times.append(np.mean(f_t))
            ivf_times.append(np.mean(i_t))

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(sizes, flat_times, "-s", color="#dc2626", label="Exact Flat Scan O(N)")
        ax3.plot(sizes, ivf_times, "-o", color="#16a34a", label="IVF-Flat (nprobe=4)")
        ax3.set_title("Graph 3: Query Latency Scaling vs. Dataset Size")
        ax3.set_xlabel("Dataset Size (Number of Vectors)")
        ax3.set_ylabel("Mean Latency (ms)")
        ax3.legend()
        ax3.grid(True)

        col_a, col_b, col_c = st.columns(3)
        col_a.pyplot(fig1)
        col_b.pyplot(fig2)
        col_c.pyplot(fig3)

with tab3:
    st.subheader("Binary Storage & CRC32 Checksum Verification")
    store_chk = load_store(5000, 32, 16, "cosine")
    ok, byte_len, crc_hex = store_chk.verify_crc32_persistence()
    st.success(f"Binary Header (`P5VS`) & CRC32 Integrity Verified: **{ok}** | Payload Size: **{byte_len:,} bytes** | Checksum: `{crc_hex}`")
