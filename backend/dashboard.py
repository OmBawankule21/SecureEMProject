import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.algorithms.em import fit_em

RESULTS = ROOT / "results"

st.set_page_config(
    page_title="SecureEM | Privacy-Preserving ML",
    page_icon="🔐",
    layout="wide"
)

st.title("SecureEM")
st.subheader("Privacy-Preserving Gaussian Mixture Model using MPC")

st.markdown(
    "Explore centralized EM, secure aggregation, numerical validation "
    "and scalability experiments."
)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "EM Experiments",
        "MPC Performance",
        "Validation"
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("MPC Parties", "3")
    c2.metric("Sufficient Statistics", "63")
    c3.metric("Backend Tests", "10/10")

    st.markdown("### Project objective")

    st.write(
        "Implement privacy-preserving distributed Gaussian Mixture "
        "Model training using Expectation-Maximization and "
        "three-party secure aggregation."
    )

    st.markdown("### Architecture")

    st.code("""
Hospital 1 ── Local E-step ──┐
Hospital 2 ── Local E-step ──┼── Fixed-point Encoding
Hospital 3 ── Local E-step ──┘
                                  |
                                  v
                         MP-SPDZ Secure Aggregation
                                  |
                                  v
                         Decode Aggregated Statistics
                                  |
                                  v
                            Global M-step
    """)

    st.info(
        "Current implementation uses three simulated parties "
        "on one machine. Independent hospital deployment is pending."
    )


# --------------------------------------------------
# EM EXPERIMENTS
# --------------------------------------------------

elif page == "EM Experiments":

    st.header("Centralized EM Experiment")

    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.write("Dataset preview")
        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        selected = st.multiselect(
            "Select numerical features",
            numeric_columns,
            default=numeric_columns
        )

        k = st.slider(
            "Number of Gaussian components",
            min_value=2,
            max_value=10,
            value=3
        )

        max_iter = st.slider(
            "Maximum EM iterations",
            min_value=10,
            max_value=200,
            value=100
        )

        if st.button("Run Centralized EM"):

            if len(selected) < 1:
                st.error("Select at least one numerical feature.")

            else:
                X = df[selected].dropna().to_numpy(dtype=float)

                if len(X) < k:
                    st.error("Dataset must contain at least k samples.")

                else:
                    # Standardize features.
                    mean = X.mean(axis=0)
                    std = X.std(axis=0)
                    std[std == 0] = 1.0
                    X = (X - mean) / std

                    with st.spinner("Running EM..."):

                        import time
                        start = time.perf_counter()

                        result = fit_em(
                            X,
                            n_components=k,
                            max_iter=max_iter
                        )

                        elapsed = time.perf_counter() - start

                    st.success("Centralized EM completed!")

                    a, b, c = st.columns(3)

                    a.metric("Runtime", f"{elapsed:.4f} s")
                    b.metric("Iterations", result["iterations"])
                    c.metric(
                        "Log-likelihood",
                        f"{result['log_likelihood']:.4f}"
                    )

                    st.markdown("### Mixture weights")
                    st.write(result["weights"])

                    st.markdown("### Cluster means")
                    st.dataframe(
                        pd.DataFrame(result["means"])
                    )

                    st.markdown("### Cluster distribution")

                    labels = pd.Series(
                        result["labels"]
                    ).value_counts().sort_index()

                    st.bar_chart(labels)


# --------------------------------------------------
# MPC PERFORMANCE
# --------------------------------------------------

elif page == "MPC Performance":

    st.header("MPC Performance and Scalability")
    st.subheader("Run Secure EM")

    sample_size = st.selectbox(
        "Select total dataset size",
        [600, 1200, 2400]
    )

    if st.button("Execute 3-Party Secure EM"):
        with st.spinner("Executing MP-SPDZ protocol..."):

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "backend" / "run_secure_em_mpc.py"),
                        str(sample_size)
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    st.success("Secure EM execution completed!")

                    st.text_area(
                        "Execution output",
                        result.stdout,
                        height=300
                    )

                    st.rerun()

                else:
                    st.error("Secure EM execution failed.")
                    st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("Execution timed out after 300 seconds.")

    # --------------------------------------------------
    # LATEST SECURE EM EXECUTION METRICS
    # --------------------------------------------------

    st.divider()
    st.subheader("Latest Secure EM Execution Metrics")

    metrics_file = RESULTS / "secure_em_metrics.json"

    if metrics_file.exists():

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Dataset Samples",
            metrics["dataset_samples"]
        )

        c2.metric(
            "EM Iterations",
            metrics["iterations"]
        )

        c3.metric(
            "Total Runtime",
            f'{metrics["total_runtime_seconds"]:.4f} s'
        )

        c4, c5, c6 = st.columns(3)

        c4.metric(
            "Peak Python Memory",
            f'{metrics["peak_memory_mb"]:.2f} MB'
        )

        c5.metric(
            "Total MPC Communication",
            f'{metrics["total_global_communication_mb"]:.6f} MB'
        )

        c6.metric(
            "Total MPC Wall Runtime",
            f'{metrics["total_mpc_wall_runtime_seconds"]:.4f} s'
        )

        st.markdown("#### Per-Iteration MPC Metrics")

        iteration_df = pd.DataFrame(
            metrics["mpc_iterations"]
        )

        iteration_df.index = iteration_df.index + 1
        iteration_df.index.name = "Iteration"

        st.dataframe(
            iteration_df,
            use_container_width=True
        )

    else:
        st.info("No Secure EM execution metrics available yet.")


    # --------------------------------------------------
    # EXPERIMENT HISTORY
    # --------------------------------------------------

    st.divider()
    st.subheader("Experiment History")

    history_file = RESULTS / "experiment_history.csv"

    if history_file.exists():

        history_df = pd.read_csv(history_file)

        st.metric(
            "Total Experiments",
            len(history_df)
        )

        st.dataframe(
            history_df.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True
        )

        st.markdown("### Historical Runtime Comparison")

        st.line_chart(
            history_df.set_index("timestamp")[
                ["total_runtime_seconds"]
            ]
        )

        st.markdown("### Dataset Size vs Runtime")

        st.scatter_chart(
            history_df,
            x="dataset_samples",
            y="total_runtime_seconds"
        )

    else:
        st.info("No experiment history available yet.")

    summary_file = RESULTS / "scalability_summary.csv"

    if summary_file.exists():

        df = pd.read_csv(summary_file)

        st.dataframe(df, use_container_width=True)
        
        st.markdown("### Mean runtime")

        st.line_chart(
            df.set_index("Samples")[["Mean_Runtime"]]
        )

        st.markdown("### Communication cost")

        st.line_chart(
            df.set_index("Samples")[["Mean_Communication"]]
        )

    else:
        st.warning("Scalability summary not found.")

    st.markdown("### Initial MPC benchmark")

    benchmark_file = RESULTS / "secure_em_benchmark.log"

    if benchmark_file.exists():

        log = benchmark_file.read_text()

        import re

        runtime = re.search(
            r"Total runtime:\s*([0-9.]+)", log
        )

        memory = re.search(
            r"Peak memory:\s*(\d+)", log
        )

        c1, c2 = st.columns(2)

        if runtime:
            c1.metric("Runtime", runtime.group(1) + " seconds")

        if memory:
            c2.metric(
                "Peak memory",
                f"{int(memory.group(1)) / 1024:.2f} MB"
            )

    st.caption(
        "Communication values represent the last MPC iteration "
        "in the current scalability summary."
    )


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

elif page == "Validation":

    st.header("Centralized vs Secure EM Validation")

    validation_file = RESULTS / "validation_metrics.csv"

    if validation_file.exists():

        df = pd.read_csv(validation_file)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.markdown("### Numerical Agreement")

        df["Absolute Difference"] = pd.to_numeric(
            df["Absolute Difference"],
            errors="coerce"
        )

        error_metrics = {
            "Iterations": "Iteration Difference",
            "Weights max error": "Weights Max Error",
            "Means max error": "Means Max Error",
            "Covariance max error": "Covariance Max Error"
        }

        col1, col2, col3, col4 = st.columns(4)

        columns = [col1, col2, col3, col4]

        for (metric, label), column in zip(
            error_metrics.items(), columns
        ):
            value = df.loc[
                df["Metric"] == metric,
                "Absolute Difference"
            ].iloc[0]

            if metric == "Iterations":
                column.metric(label, int(value))
            else:
                column.metric(label, f"{value:.2e}")

        error_df = df[
            df["Metric"].isin([
                "Weights max error",
                "Means max error",
                "Covariance max error"
            ])
        ].copy()

        st.markdown("### Parameter-wise Numerical Error")

        st.bar_chart(
            error_df.set_index("Metric")["Absolute Difference"]
        )

        st.success(
            "Previously recorded numerical validation results loaded."
        )

    else:
        st.warning("Validation metrics file not found.")

    # --------------------------------------------------
    # CENTRALIZED VS SECURE EM RUNTIME COMPARISON
    # --------------------------------------------------

    st.divider()
    st.subheader("Centralized EM vs Secure EM Runtime")

    runtime_file = RESULTS / "runtime_comparison.csv"

    if runtime_file.exists():

        runtime_df = pd.read_csv(runtime_file)

        st.dataframe(
            runtime_df,
            use_container_width=True
        )

        st.markdown("### Execution Time Comparison")

        chart_df = runtime_df.set_index("Algorithm")

        st.bar_chart(
            chart_df["Runtime_seconds"]
        )

        central_time = runtime_df.loc[
            runtime_df["Algorithm"] == "Centralized EM",
            "Runtime_seconds"
        ].iloc[0]

        secure_time = runtime_df.loc[
            runtime_df["Algorithm"] == "Secure EM",
            "Runtime_seconds"
        ].iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "Centralized EM",
            f"{central_time:.5f} sec"
        )

        col2.metric(
            "Secure EM",
            f"{secure_time:.5f} sec"
        )

        st.metric(
            "Secure EM Slowdown",
            f"{secure_time / central_time:.2f}x"
        )

    else:
        st.warning("Runtime comparison file not found.")


    # --------------------------------------------------
    # REPEATED PERFORMANCE BENCHMARK
    # --------------------------------------------------

    st.divider()
    st.subheader("Repeated Performance Benchmark")

    repeated_file = RESULTS / "repeated_comparison_summary.csv"

    if repeated_file.exists():

        repeated_df = pd.read_csv(repeated_file)

        st.markdown("### Benchmark Summary (5 Trials)")

        st.dataframe(
            repeated_df,
            use_container_width=True
        )

        # Runtime comparison
        st.markdown("### Mean Runtime Comparison")

        runtime_pivot = repeated_df.pivot(
            index="Samples",
            columns="Algorithm",
            values="Mean_Runtime"
        )

        st.line_chart(runtime_pivot)

        # Standard deviation
        st.markdown("### Runtime Standard Deviation")

        std_pivot = repeated_df.pivot(
            index="Samples",
            columns="Algorithm",
            values="Std_Runtime"
        )

        st.line_chart(std_pivot)

        # Communication cost
        st.markdown("### MPC Communication Cost")

        communication_df = repeated_df[
            repeated_df["Algorithm"] == "Secure EM"
        ]

        st.bar_chart(
            communication_df.set_index("Samples")[
                "Mean_Communication_MB"
            ]
        )

        # Slowdown calculation
        st.markdown("### Secure EM Slowdown")

        slowdown_df = runtime_pivot.copy()

        slowdown_df["Slowdown"] = (
            slowdown_df["Secure EM"] /
            slowdown_df["Centralized EM"]
        )

        st.dataframe(
            slowdown_df[["Slowdown"]].round(2),
            use_container_width=True
        )

        st.line_chart(slowdown_df[["Slowdown"]])

        st.success(
            "Repeated benchmark results loaded successfully."
        )

    else:
        st.warning("Repeated comparison summary not found.")


    secure_file = RESULTS / "secure_em_results.npz"

    if secure_file.exists():

        data = np.load(secure_file)

        st.markdown("### Secure EM mixture weights")
        st.write(data["weights"])

        st.markdown("### Secure EM means")
        st.dataframe(pd.DataFrame(data["means"]))

        
        st.markdown("### Secure EM covariance matrices")

        for i, covariance in enumerate(data["covariances"]):
            st.markdown(f"**Component {i + 1}**")
            st.dataframe(
                pd.DataFrame(
                    covariance,
                    columns=[f"Feature {j+1}" for j in range(covariance.shape[1])],
                    index=[f"Feature {j+1}" for j in range(covariance.shape[0])]
                )
            )
        


    st.info(
        "The current MPC implementation uses simulated parties "
        "on one machine. This dashboard does not represent "
        "independent hospital deployment."
    )

st.divider()

st.caption(
    "SecureEM | Research Prototype | MP-SPDZ + Python + Streamlit"
)
