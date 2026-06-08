import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_css, render_sidebar

inject_css()
render_sidebar()

st.set_page_config(layout="wide")
st.markdown('<div class="page-header"><h2>Model Performance</h2><p>Compare evaluation metrics and visual diagnostics across all four sentiment models.</p></div>', unsafe_allow_html=True)

MODELS = ["Naive Bayes", "SVM", "LSTM", "IndoBERT"]
MODEL_KEYS = ["nb", "svm", "lstm", "indobert"]

METRICS = {
    "Accuracy":    {"nb": 0.9397, "svm": 0.9698, "lstm": 0.9310, "indobert": 0.9720},
    "Precision":   {"nb": 0.9403, "svm": 0.9706, "lstm": 0.9311, "indobert": 0.9720},
    "Recall":      {"nb": 0.9397, "svm": 0.9698, "lstm": 0.9310, "indobert": 0.9720},
    "F1-Score":    {"nb": 0.9395, "svm": 0.9698, "lstm": 0.9311, "indobert": 0.9720},
    "Specificity": {"nb": 0.9634, "svm": 0.9919, "lstm": 0.9309, "indobert": 0.9797}
}

METRIC_ICONS = {"Accuracy": "🎯", "Precision": "📐", "Recall": "📡", "F1-Score": "⚖️", "Specificity": "🛡️"}

st.markdown("<h3 style='color: var(--text-main);'>Metric Cards</h3>", unsafe_allow_html=True)
st.markdown("")

for metric_name, metric_vals in METRICS.items():
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem; margin-top:1.25rem;">
        <span style="font-size:1.1rem">{METRIC_ICONS[metric_name]}</span>
        <span style="font-family:'Sora',sans-serif; font-weight:700; font-size:1rem; color:var(--text-main);">{metric_name}</span>
        <div style="flex:1; height:1px; background:var(--border); margin-left:0.5rem;"></div>
    </div>
    """, unsafe_allow_html=True)
    max_val = max(metric_vals.values())
    cols = st.columns(4, gap="small")
    for i, (model, key) in enumerate(zip(MODELS, MODEL_KEYS)):
        val = metric_vals[key]
        display = f"{val:.2%}" if val > 0 else "—"
        is_highest = (val == max_val)
        if is_highest:
            highlight_style = "background-color: var(--primary-mid);"
        else:
            highlight_style = ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="{highlight_style}">
                <div class="label">{model}</div>
                <div class="value" style="font-size:1.5rem">{display}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<h3 style='color: var(--text-main);'>Evaluation Metric Plots</h3>", unsafe_allow_html=True)
st.markdown("")

PLOT_GROUPS = [
    {"model": "Naive Bayes", "plots": [("Confusion Matrix", "naive_bayes_confusion_matrix"), ("ROC-PR Curves", "naive_bayes_roc_pr_curves"), ("Train vs Val Metrics", "naive_bayes_train_vs_val_metrics")]},
    {"model": "SVM",         "plots": [("Confusion Matrix", "svm_confusion_matrix"),         ("ROC-PR Curves", "svm_roc_pr_curves"),          ("Train vs Val Metrics", "svm_train_vs_val_metrics")]},
    {"model": "LSTM",        "plots": [("Confusion Matrix", "lstm_confusion_matrix"),        ("ROC-PR Curves", "lstm_roc_pr_curves"),         ("Train vs Val Loss", "lstm_loss")]},
    {"model": "IndoBERT",    "plots": [("Confusion Matrix", "indobert_confusion_matrix"),    ("ROC-PR Curves", "indobert_roc_pr_curves"),     ("Train vs Val Loss", "indobert_loss")]},
]

for group in PLOT_GROUPS:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem; margin-top:1.5rem;">
        <span style="font-family:'Sora',sans-serif; font-weight:700; font-size:1rem; color:var(--text-main);">{group["model"]}</span>
        <div style="flex:1; height:1px; background:var(--border); margin-left:0.5rem;"></div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(3, gap="small")
    for i, (plot_label, plot_key) in enumerate(group["plots"]):
        asset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", f"{plot_key}.png")
        with cols[i]:
            if os.path.exists(asset_path):
                st.image(asset_path, caption=plot_label, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="plot-placeholder">
                    <div class="ph-icon">📊</div>
                    <div class="ph-label">{plot_label}<br><span style="font-weight:400;opacity:0.7;">Place image at<br>assets/{plot_key}.png</span></div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)