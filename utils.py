import streamlit as st

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

:root {
    --primary: #E8530A;
    --primary-light: #FF6B2B;
    --primary-pale: #FFF0EA;
    --primary-mid: #FDDDD0;
    --accent: #1A1A2E;
    --text-main: #1C1C1C;
    --text-muted: #6B7280;
    --text-light: #9CA3AF;
    --bg-base: #FAFAF8;
    --bg-card: #FFFFFF;
    --border: #E5E7EB;
    --border-strong: #D1D5DB;
    --positive: #059669;
    --positive-bg: #ECFDF5;
    --negative: #DC2626;
    --negative-bg: #FEF2F2;
    --radius: 12px;
    --radius-lg: 18px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 30px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.06);
}

.stApp { background-color: var(--bg-base); }

section[data-testid="stSidebar"] {
    background-color: var(--accent);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] p { color: #9CA3AF !important; }
section[data-testid="stSidebar"] a { color: #E5E7EB !important; text-decoration: none; }
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background-color: rgba(232, 83, 10, 0.2) !important;
    border-left: 3px solid var(--primary) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebarNav"] a:hover { background-color: rgba(255,255,255,0.08) !important; }

h1, h2, h3, h4, h5 { font-family: 'Sora', sans-serif; color: var(--text-main); }

.stButton > button {
    background-color: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(232, 83, 10, 0.30) !important;
}
.stButton > button:hover {
    background-color: var(--primary-light) !important;
    box-shadow: 0 4px 14px rgba(232, 83, 10, 0.40) !important;
    transform: translateY(-1px) !important;
}

.stTextArea textarea {
    border: 1.5px solid var(--border-strong) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text-main) !important;
    background: var(--bg-card) !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(232, 83, 10, 0.12) !important;
}

.stFileUploader {
    border: 2px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-card) !important;
}
.stFileUploader:hover { border-color: var(--primary) !important; }

.stSelectbox > div > div {
    border: 1.5px solid var(--border-strong) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
}

.stTab [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTab [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}
.stTab [aria-selected="true"] { background-color: var(--primary) !important; color: white !important; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.metric-card .label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--text-light); margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: 'Sora', sans-serif; font-size: 2rem;
    font-weight: 800; color: var(--text-main); line-height: 1;
}

.result-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow); margin-bottom: 0.75rem;
}
.result-card .model-tag {
    display: inline-block; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--primary); background: var(--primary-pale);
    border-radius: 6px; padding: 2px 8px; margin-bottom: 0.6rem;
}
.result-card .sentiment-label { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; }
.sentiment-pos { color: var(--positive); }
.sentiment-neg { color: var(--negative); }

.confidence-bar-wrap {
    height: 8px; background: var(--border); border-radius: 99px;
    overflow: hidden; margin-top: 0.5rem;
}
.confidence-bar { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.bar-pos { background: var(--positive); }
.bar-neg { background: var(--negative); }

.phrase-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.7rem 0; border-bottom: 1px solid var(--border);
}
.phrase-row:last-child { border-bottom: none; }
.phrase-text { min-width: 180px; font-weight: 500; color: var(--text-main); font-size: 0.9rem; }
.phrase-bar-wrap { flex: 1; height: 10px; background: var(--border); border-radius: 99px; overflow: hidden; }
.phrase-bar-pos { background: linear-gradient(90deg, #10B981, #059669); height: 100%; border-radius: 99px; }
.phrase-bar-neg { background: linear-gradient(90deg, #F87171, #DC2626); height: 100%; border-radius: 99px; }
.phrase-count { min-width: 80px; text-align: right; font-size: 0.82rem; color: var(--text-muted); font-weight: 500; }

.section-divider { height: 1px; background: var(--border); margin: 2rem 0; }

.pipeline-step {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 4px solid var(--primary); border-radius: var(--radius);
    padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: var(--shadow);
}
.pipeline-step .step-num {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--primary); margin-bottom: 0.25rem;
}
.pipeline-step .step-title {
    font-family: 'Sora', sans-serif; font-weight: 700;
    font-size: 1rem; color: var(--text-main); margin-bottom: 0.25rem;
}
.pipeline-step .step-desc { font-size: 0.87rem; color: var(--text-muted); line-height: 1.5; }

.plot-placeholder {
    background: var(--bg-card); border: 2px dashed var(--border-strong);
    border-radius: var(--radius-lg); display: flex; flex-direction: column;
    align-items: center; justify-content: center; min-height: 220px; gap: 0.5rem;
}
.plot-placeholder .ph-icon { font-size: 2rem; opacity: 0.35; }
.plot-placeholder .ph-label { font-size: 0.82rem; font-weight: 600; color: var(--text-light); text-align: center; }

.overview-stat {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.5rem; text-align: center; box-shadow: var(--shadow);
}
.overview-stat .ov-pct {
    font-family: 'Sora', sans-serif; font-size: 3rem;
    font-weight: 800; line-height: 1; margin-bottom: 0.4rem;
}
.overview-stat .ov-label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }

.stMarkdown p { color: var(--text-main); line-height: 1.65; }

.page-header { margin-bottom: 2.5rem; }
.page-header h2 {
    font-family: 'Sora', sans-serif; font-size: 1.9rem;
    font-weight: 800; color: var(--text-main); margin-bottom: 0.4rem;
}
.page-header p { font-size: 1rem; color: var(--text-muted); }

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
"""

def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0.5rem 1rem;">
            <div style="font-family:'Sora',sans-serif; font-size:1.4rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">
                SentiPhrase
            </div>
            <div style="font-size:0.75rem; color:#6B7280; font-weight:500; margin-top:2px; text-transform:uppercase; letter-spacing:0.1em;">
                Sentiment Analysis
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.08); margin:1rem 0;"></div>
        </div>
        """, unsafe_allow_html=True)
