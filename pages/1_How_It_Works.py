import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_css, render_sidebar

inject_css()
render_sidebar()

st.set_page_config(layout="wide")
st.markdown('<div class="page-header"><h2>How It Works</h2><p>Explore the general pipeline and how each model approaches phrase-level sentiment analysis.</p></div>', unsafe_allow_html=True)

st.markdown("<h3 style='color: var(--text-main);'>General Pipeline</h3>", unsafe_allow_html=True)
st.markdown("")

pipeline_steps = [
    ("01", "Data Loading & Preprocessing", "Raw CSV reviews are loaded and cleaned: URLs removed, text lowercased, slang normalized via a curated Indonesian slang dictionary, repeated characters collapsed, and stopwords filtered."),
    ("02", "Sentence Splitting", "Each review is split into clausal phrases using a set of conjunctions and punctuation rules (e.g. splitting on <em>tapi, tetapi, namun, meski</em>). Negation words are preserved and prepended to the next clause."),
    ("03", "Phrase Filtering", "Phrases are filtered by minimum token count, character length, and word validity (alphabetic, min 3 chars). Near-duplicate phrases are removed using cosine similarity thresholding (≥ 0.99)."),
    ("04", "Feature Extraction / Embedding", "Each model transforms phrases into numerical representations: TF-IDF vectors for classical models, FastText word embeddings for LSTM, and WordPiece tokenization for IndoBERT."),
    ("05", "Sentiment Classification", "The model classifies each phrase as <strong>Positif</strong> or <strong>Negatif</strong>. Phrases below the confidence threshold are discarded."),
    ("06", "Aggregation & Summary", "Classified phrases are deduplicated and grouped. Frequency counts and percentages are computed. Results are summarized as a ranked phrase list with sentiment labels."),
]

for num, title, desc in pipeline_steps:
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="step-num">Step {num}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("<h3 style='color: var(--text-main);'>Model Pipeline Comparison</h3>", unsafe_allow_html=True)
st.markdown("")

st.markdown("""
<style>
.cmp-table { width:100%; table-layout: fixed; border-collapse:collapse; font-family:'DM Sans',sans-serif; font-size:0.88rem; border-radius:14px; overflow:hidden; box-shadow:var(--shadow-md); background:var(--bg-card); }
.cmp-table thead tr { background:var(--accent); }
.cmp-table thead th { color:#FFFFFF; font-family:'Sora',sans-serif; font-weight:700; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.07em; padding:1rem 1.25rem; text-align:left; border:none; text-align: center; }
.cmp-table thead th:first-child { border-radius:14px 0 0 0; min-width:160px; }
.cmp-table thead th:last-child { border-radius:0 14px 0 0; }
.cmp-table tbody tr { border-bottom:1px solid var(--border); transition:background 0.15s ease; }
.cmp-table tbody tr:hover { background:var(--primary-pale); }
.cmp-table tbody tr:last-child { border-bottom:none; }
.cmp-table td { padding:0.9rem 1.25rem; vertical-align:center; color:var(--text-main); line-height:1.5; text-align: center; }
.cmp-table td:first-child { font-family:'Sora',sans-serif; font-weight:700; color:var(--primary); white-space:nowrap; }
.cmp-table .tag { display:inline-block; background:var(--primary-pale); color:var(--primary); border-radius:6px; padding:1px 7px; font-size:0.75rem; font-weight:700; margin-right:2px; }
.cmp-table .tag-green { background:var(--positive-bg); color:var(--positive); }
.cmp-table .tag-blue { background:#EFF6FF; color:#2563EB; }
</style>
<table class="cmp-table">
    <thead>
        <tr><th>Stage</th><th>Naive Bayes</th><th>SVM</th><th>LSTM</th><th>IndoBERT</th></tr>
    </thead>
    <tbody>
        <tr>
            <td>Model Type</td>
            <td>Supervised ML<br>Probabilistic</td>
            <td>Supervised ML<br>Discriminative</td>
            <td>Supervised DL<br>RNN</td>
            <td>LLM<br>Transformer</td>
        </tr>
        <tr>
            <td>Data Loading</td>
            <td colspan="4">Rename columns -> filter ≥ 5 words -> drop duplicates -> binary label</td>
        </tr>
        <tr>
            <td>Text Normalization</td>
            <td>Lowercase + regex + slang norm + lemmatize + stopword removal</td>
            <td>Lowercase + regex + slang norm</td>
            <td>Lowercase + regex + slang norm</td>
            <td>Lowercase + regex + slang norm</td>
        </tr>
        <tr>
            <td>Feature Extraction</td>
            <td>TF-IDF Vectorizer</td>
            <td>TF-IDF Vectorizer</td>
            <td>FastText pretrained word embeddings</td>
            <td>WordPiece tokenization<br><span class="tag tag-blue">indobert-base-p1</span></td>
        </tr>
        <tr>
            <td>Model Architecture</td>
            <td>Multinomial Naive Bayes<br><span class="tag">No epochs</span></td>
            <td>Support Vector Classifier<br><span class="tag">No epochs</span></td>
            <td>Long-Short Term Memory<br><span class="tag">epochs=5</span></td>
            <td>IndoBERT<br><span class="tag">epochs=3</span></td>
        </tr>
        <tr>
            <td>Lexicon Extraction</td>
            <td>Logarithm Probability Difference</td>
            <td>Coefficient Weights</td>
            <td>Captum Integrated Gradient</td>
            <td>Captum Integrated Gradient</td>
        </tr>
        <tr>
            <td>Phrase Extraction</td>
            <td colspan="4">Clause splitting on conjunctions and punctuation<br>POS tagging and dependency parsing using Stanza<br>Rule-based phrase extraction</td>
        </tr>
        <tr>
            <td>Semantic Deduplication</td>
            <td>Cosine similarity ≥ 0.99 on TF-IDF</td>
            <td>Cosine similarity ≥ 0.99 on TF-IDF</td>
            <td>Cosine similarity ≥ 0.99 on FastText vectors</td>
            <td>Cosine similarity ≥ 0.99 on IndoBERT embeddings</td>
        </tr>
        <tr>
            <td>Labels</td>
            <td colspan="4" style="text-align:center; color:var(--text-muted); font-style:italic;">
                <strong style="color:var(--positive)">Positif</strong> &nbsp;|&nbsp; <strong style="color:var(--negative)">Negatif</strong> &nbsp;-&nbsp; binary classification across all four models
            </td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:var(--primary-pale); border:1px solid var(--primary-mid); border-radius:var(--radius); padding:1rem 1.25rem; font-size:0.88rem; color:var(--text-main); line-height:1.6;">
    <strong style="color:var(--primary);">Dataset:</strong> PRDECT-ID - a publicly available Indonesian e-commerce product review dataset with sentiment annotations (Positif / Negatif).
    All four models share the same preprocessing pipeline up to the feature extraction stage.
</div>
""", unsafe_allow_html=True)
