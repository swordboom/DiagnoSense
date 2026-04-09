import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"


def _load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_pct(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = (df[col].astype(float) * 100).map("{:.2f}%".format)
    return df


def render_comparison_page():
    st.title("Model Comparison")
    st.markdown("---")

    metrics = _load_metrics()
    if metrics is None:
        st.warning("Metrics not found yet. Run Phase 5 evaluation to populate this page.")
        return

    health = metrics["health"]
    medicine = metrics["medicine"]
    split_meta = metrics.get("split_metadata", {})
    comparison = metrics.get("comparison", {})
    h_diag = health.get("diagnostics", {})
    m_diag = medicine.get("diagnostics", {})

    st.subheader("DiagnoSense Model Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Health (DiagnoSense-MLP)**")
        st.metric("Test Accuracy", f"{health['test_accuracy'] * 100:.2f}%")
        st.metric("Macro F1", f"{health['macro_f1'] * 100:.2f}%")
        st.metric("Macro Precision", f"{health['macro_precision'] * 100:.2f}%")
        st.metric("Macro Recall", f"{health['macro_recall'] * 100:.2f}%")
        st.caption(f"Test samples: {health['test_samples']}")

    with col2:
        st.markdown("**Medicine (DiagnoSense-MLP)**")
        st.metric("Micro F1", f"{medicine['micro_f1'] * 100:.2f}%")
        st.metric("Samples F1", f"{medicine['samples_f1'] * 100:.2f}%")
        st.metric("Macro F1", f"{medicine['macro_f1'] * 100:.2f}%")
        st.metric("Threshold", f"{medicine['threshold']:.2f}")
        st.caption(f"Test samples: {medicine['test_samples']} | Labels: {medicine['num_labels']}")

    st.subheader("Benchmark Against Standard Models")
    health_rows = comparison.get("health", [])
    medicine_rows = comparison.get("medicine", [])

    if health_rows:
        st.markdown("**Health (same split, same features)**")
        health_df = pd.DataFrame(health_rows)
        health_df = _format_pct(health_df, ["accuracy", "macro_f1", "macro_precision", "macro_recall"])
        if "fit_eval_seconds" in health_df.columns:
            health_df = health_df.drop(columns=["fit_eval_seconds"])
        st.dataframe(health_df, use_container_width=True, hide_index=True)
    else:
        st.info("Health benchmark table not available yet.")

    if medicine_rows:
        st.markdown("**Medicine (same split, same features)**")
        med_df = pd.DataFrame(medicine_rows)
        med_df = _format_pct(med_df, ["micro_f1", "macro_f1", "samples_f1", "micro_precision", "micro_recall"])
        if "fit_eval_seconds" in med_df.columns:
            med_df = med_df.drop(columns=["fit_eval_seconds"])
        if "threshold" in med_df.columns:
            med_df["threshold"] = med_df["threshold"].astype(float).map("{:.2f}".format)
        st.dataframe(med_df, use_container_width=True, hide_index=True)
    else:
        st.info("Medicine benchmark table not available yet.")

    st.subheader("Overfitting and Bias Diagnostics")
    if h_diag and m_diag:
        diag_table = [
            {
                "Dataset": "Health",
                "Overfitting risk": h_diag.get("overfitting_risk", "N/A"),
                "Train-Test gap": f"{h_diag.get('train_test_gap', {}).get('macro_f1', 0.0):.4f} (macro-F1)",
                "Bias risk proxy": h_diag.get("class_balance_bias_risk", "N/A"),
            },
            {
                "Dataset": "Medicine",
                "Overfitting risk": m_diag.get("overfitting_risk", "N/A"),
                "Train-Test gap": f"{m_diag.get('train_test_gap', {}).get('micro_f1', 0.0):.4f} (micro-F1)",
                "Bias risk proxy": m_diag.get("label_balance_bias_risk", "N/A"),
            },
        ]
        st.table(diag_table)
        st.caption(
            "Bias risk proxy is based on recall disparity across classes/labels because demographic attributes are not present."
        )
        if h_diag.get("dataset_realism_warning", False):
            st.warning(
                "Health dataset may still be over-separable: multiple standard models are scoring near-perfect on the same split."
            )
    else:
        st.info("Diagnostics are not available yet. Re-run Phase 5.")

    st.subheader("Leakage Checks")
    if split_meta:
        h_meta = split_meta.get("health", {})
        m_meta = split_meta.get("medicine", {})
        leakage_data = [
            {
                "Dataset": "Health",
                "Train/Val overlap": h_meta.get("text_overlap_train_val", "N/A"),
                "Train/Test overlap": h_meta.get("text_overlap_train_test", "N/A"),
                "Val/Test overlap": h_meta.get("text_overlap_val_test", "N/A"),
            },
            {
                "Dataset": "Medicine",
                "Train/Val overlap": m_meta.get("text_overlap_train_val", "N/A"),
                "Train/Test overlap": m_meta.get("text_overlap_train_test", "N/A"),
                "Val/Test overlap": m_meta.get("text_overlap_val_test", "N/A"),
            },
        ]
        st.table(leakage_data)
    else:
        st.info("Split metadata is unavailable.")

