import streamlit as st
from utils import inject_css, render_sidebar

st.set_page_config(
    page_title="SentiPhrase",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar()

st.markdown("""
<style>
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}
.home-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 80vh; text-align: center; padding: 2rem 1rem;
}
.home-site-name {
    font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.18em; color: var(--primary); margin-bottom: 1.5rem;
}
.home-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--primary-pale); color: var(--primary); border-radius: 99px;
    padding: 5px 14px; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 1.5rem;
}
.home-main-title {
    font-family: 'Sora', sans-serif; font-size: clamp(2.8rem, 6vw, 4.8rem);
    font-weight: 800; color: var(--text-main); line-height: 1.08;
    margin-bottom: 1.5rem; max-width: 820px;
}
.home-main-title .highlight { color: var(--primary); }
.home-desc {
    font-size: 1.1rem; color: var(--text-muted); max-width: 520px;
    line-height: 1.65; margin-bottom: 3rem;
}
.home-models-row {
    display: flex; gap: 0.75rem; justify-content: center;
    flex-wrap: wrap; margin-bottom: 3.5rem;
}
.model-chip {
    background: var(--bg-card); border: 1.5px solid var(--border);
    border-radius: 99px; padding: 6px 16px; font-size: 0.82rem;
    font-weight: 600; color: var(--text-main); box-shadow: var(--shadow);
}
.home-divider {
    width: 64px; height: 3px; background: var(--primary);
    border-radius: 99px; margin: 0 auto 3rem;
}
.home-stats-row { display: flex; gap: 3rem; justify-content: center; flex-wrap: wrap; }
.home-stat { text-align: center; }
.home-stat .s-val {
    font-family: 'Sora', sans-serif; font-size: 2.2rem;
    font-weight: 800; color: var(--text-main); line-height: 1;
}
.home-stat .s-lbl {
    font-size: 0.8rem; color: var(--text-muted); font-weight: 500;
    margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.06em;
}
</style>

<div class="home-wrap">
    <div class="home-badge">📦 Indonesian Marketplace Reviews</div>
    <div class="home-site-name">SentiPhrase</div>
    <div class="home-main-title">
        Phrase-Level <span class="highlight">Sentiment Analysis</span><br>for Product Reviews
    </div>
    <p class="home-desc">
        Detect positive and negative sentiments in individual phrases extracted from Indonesian online marketplace product reviews using four distinct AI models.
    </p>
    <div class="home-models-row">
        <div class="model-chip">Naive Bayes</div>
        <div class="model-chip">SVM</div>
        <div class="model-chip">LSTM</div>
        <div class="model-chip">IndoBERT</div>
    </div>
    <div class="home-divider"></div>
    <div class="home-stats-row">
        <div class="home-stat"><div class="s-val">4</div><div class="s-lbl">Models</div></div>
        <div class="home-stat"><div class="s-val">2</div><div class="s-lbl">Sentiments</div></div>
        <div class="home-stat"><div class="s-val">ID</div><div class="s-lbl">Language</div></div>
        <div class="home-stat"><div class="s-val">∞</div><div class="s-lbl">Reviews</div></div>
    </div>
</div>
""", unsafe_allow_html=True)
