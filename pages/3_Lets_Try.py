import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import joblib
import sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_css, render_sidebar

inject_css()
render_sidebar()

st.set_page_config(layout="wide")
st.markdown('<div class="page-header"><h2>Let\'s Try</h2><p>Run phrase-level sentiment analysis on your own review text or batch files.</p></div>', unsafe_allow_html=True)

HF_NB_REPO       = "venesia-ari/naive-bayes-phrase-level-sentiment-analysis"
HF_SVM_REPO      = "venesia-ari/svm-phrase-level-sentiment-analysis"
HF_LSTM_REPO     = "venesia-ari/lstm-phrase-level-sentiment-analysis"
HF_INDOBERT_REPO = "venesia-ari/indobert-phrase-level-sentiment-analysis"
HF_TOKEN         = st.secrets["HF_TOKEN"]

MODELS_CONFIG = {
    "Naive Bayes": {"key": "nb",       "repo": HF_NB_REPO,       "type": "nb",       "threshold_pos": 0.75, "threshold_neg": 0.75},
    "SVM":         {"key": "svm",      "repo": HF_SVM_REPO,      "type": "svm",      "threshold": 0.75},
    "LSTM":        {"key": "lstm",     "repo": HF_LSTM_REPO,     "type": "lstm",     "threshold": 0.50},
    "IndoBERT":    {"key": "indobert", "repo": HF_INDOBERT_REPO, "type": "indobert", "threshold": 0.75},
}

NEGATION_WORDS     = {"tidak", "bukan", "belum", "jangan", "kurang", "tanpa"}
SPLIT_CONJUNCTIONS = ["tapi", "tetapi", "namun", "meski", "meskipun", "walaupun", "walau", "dan", "serta", "juga", "karena", "soalnya", "sebab", "padahal", "sedangkan"]

STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "adalah", "dari", "dalam", "ke", "akan", "oleh", "saya", "aku", "kamu", "dia", "kami",
    "mereka", "nya", "ada", "sudah", "belum", "juga", "bisa", "hanya", "lebih", "lagi", "sangat", "sekali", "kalau", "jika", "atau", "karena", "tapi",
    "tetapi", "namun", "se", "si", "pun", "lah", "kah", "dong", "deh", "sih", "nih", "ya", "yah", "kok", "kan", "mau", "mah", "banget", "aja",
    "udah", "jadi", "sama", "satu", "dua", "tiga", "masih", "saat", "waktu", "baru", "harus", "banyak", "lain", "kali",
}

SIMILARITY_THRESHOLD = 0.99

REGEX_PATTERNS = [
    (r"(.)\1{2,}",          r"\1\1"),
    (r"([a-zA-Z])\1\b",     r"\1"),
    (r"([!?,;.]){2,}",      r"\1"),
    (r"\b(\w+)\s+\1\b",     r"\1"),
    (r"\s+([!?,;.])",       r"\1"),
    (r"([!?,;.])(?!\s)",    r"\1 "),
]

SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "engga": "tidak", "tdk": "tidak", "tak": "tidak",
    "tp": "tapi", "tpi": "tapi", "cuma": "hanya", "cmn": "hanya", "cuman": "hanya",
    "bgt": "banget", "bgd": "banget", "bngt": "banget", "bet": "banget", "bnget": "banget", "sgt": "sangat", "sngat": "sangat", "sangt": "sangat",
    "sy": "saya", "sya": "saya", "gw": "saya", "gue": "saya", "aku": "saya", "ak": "saya", "lo": "kamu", "lu": "kamu", "kmu": "kamu",
    "ud": "sudah", "udh": "sudah", "uda": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "blum": "belum", "belom": "belum", "blom": "belum",
    "lg": "lagi", "lgi": "lagi", "bs": "bisa", "bsa": "bisa", "skrg": "sekarang", "skr": "sekarang", "skg": "sekarang", "cb": "coba", "cba": "coba",
    "br": "baru", "bru": "baru", "cpt": "cepat", "cepet": "cepat", "cpet": "cepat", "lm": "lama", "lma": "lama", "hrs": "harus", "hrus": "harus",
    "bgs": "bagus", "bgus": "bagus", "mntp": "bagus", "mntap": "bagus", "mantep": "bagus", "mntep": "bagus", "mantap": "bagus",
    "sj": "saja", "aja": "saja", "sja": "saja", "aj": "saja", "doank": "saja", "doang": "saja",
    "bbrp": "beberapa", "bbrapa": "beberapa", "bbrpa": "beberapa", "bebrpa": "beberapa", "brp": "berapa", "brpa": "berapa", "brapa": "berapa",
    "bgini": "seperti ini", "begini": "seperti ini", "sperti": "seperti", "yg": "yang", "yng": "yang", "krn": "karena", "karna": "karena",
    "dgn": "dengan", "dg": "dengan", "dr": "dari", "dri": "dari", "utk": "untuk", "buat": "untuk", "jg": "juga", "sm": "sama", "mk": "maka",
    "brg": "barang", "brng": "barang", "ongkir": "ongkos kirim", "ori": "original", "ok": "oke", "pngiriman": "pengiriman", "pngirim": "pengirim", "krm": "kirim",
    "saller": "penjual", "saler": "penjual", "seler": "penjual", "seller": "penjual", "pnjual": "penjual",
    "lmyn": "lumayan", "lmayan": "lumayan", "kualits": "kualitas", "packing": "packaging",
    "wkwk": "", "wkwkwk": "", "haha": "", "hihi": "", "hehe": "", "masyaallah": "", "sih": "", "alhamdulilah": "", "alhamdullilah": "", "alhamdulillah": "",
    "mantul": "bagus banget", "gokil": "luar biasa", "okesip": "oke siap",
}

ID2LABEL = {0: "negatif", 1: "positif"}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MAX_LENGTH = 128
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2

def apply_regex_patterns(text):
    for pattern, replacement in REGEX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    words = text.split()
    normalized = []
    for word in words:
        replacement = SLANG_DICT.get(word, word)
        if replacement:
            normalized.extend(replacement.split())
    text = " ".join(normalized)
    text = apply_regex_patterns(text)
    text = re.sub(r"[^\w\s,.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_clauses(text):
    if not isinstance(text, str) or not text.strip():
        return []
    conj_pattern   = r"\b(?:" + "|".join(map(re.escape, SPLIT_CONJUNCTIONS)) + r")\b"
    processed_text = re.sub(conj_pattern, "|", text, flags=re.IGNORECASE)
    processed_text = re.sub(r"[.,!?;\s*]{2,}|[.,!?;]", "|", processed_text)
    raw_clauses    = processed_text.split("|")
    all_clauses    = []
    for clause in raw_clauses:
        cleaned = re.sub(r"\s+", " ", clause).strip()
        if cleaned:
            all_clauses.append(cleaned)
    return all_clauses

def pos_and_dep_parse(text, nlp_stanza):
    if not isinstance(text, str) or not text.strip():
        return []
    doc    = nlp_stanza(text)
    tokens = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append({
                "id"     : word.id - 1,
                "word"   : word.lemma.lower() if word.lemma else word.text.lower(),
                "pos"    : word.upos,
                "dep"    : word.deprel,
                "head_id": word.head - 1,
            })
    return tokens

def extract_phrases(clause, nlp_stanza):
    tokens = pos_and_dep_parse(clause, nlp_stanza)
    if not tokens:
        return []
    phrases  = []
    seen     = set()
    n        = len(tokens)
    NOUN_POS = {"NOUN", "PROPN"}
    ADJ_POS  = {"ADJ"}
    ADV_POS  = {"ADV"}
    VERB_POS = {"VERB"}
    def is_neg(tok):
        return tok["word"] in NEGATION_WORDS
    def add(tokens_in_phrase, pattern):
        text = " ".join(t["word"] for t in tokens_in_phrase)
        if text not in seen:
            seen.add(text)
            phrases.append({"phrase": text, "pattern": pattern})
    for i in range(n - 1):
        if tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in ADJ_POS:
            if not is_neg(tokens[i]) and not is_neg(tokens[i+1]):
                add([tokens[i], tokens[i+1]], "R1_NOUN+ADJ")
    for i in range(n - 2):
        if (tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in ADV_POS and tokens[i+2]["pos"] in ADJ_POS and not is_neg(tokens[i+1])):
            add([tokens[i], tokens[i+1], tokens[i+2]], "R2_NOUN+ADV+ADJ")
    for i in range(n - 2):
        if (tokens[i]["pos"] in NOUN_POS and is_neg(tokens[i+1]) and tokens[i+2]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2]], "R3_NOUN+NEG+ADJ")
    for i in range(n - 3):
        if (tokens[i]["pos"] in NOUN_POS and is_neg(tokens[i+1]) and tokens[i+2]["pos"] in ADV_POS and tokens[i+3]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]], "R4_NOUN+NEG+ADV+ADJ")
    for i in range(n - 1):
        if tokens[i]["pos"] in ADJ_POS and tokens[i+1]["pos"] in NOUN_POS:
            if tokens[i+1]["dep"] in {"root", "nsubj"}:
                add([tokens[i+1], tokens[i]], "R5_ADJ+NOUN")
    for i in range(n - 2):
        if (tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in VERB_POS and tokens[i+2]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2]], "R6_NOUN+VERB+ADJ")
    for i in range(n - 3):
        if (tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in VERB_POS and is_neg(tokens[i+2]) and tokens[i+3]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]], "R7_NOUN+VERB+NEG+ADJ")
    for i in range(n - 2):
        if (tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in NOUN_POS and tokens[i+2]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2]], "R8_NOUN+NOUN+ADJ")
    for i in range(n - 3):
        if (tokens[i]["pos"] in NOUN_POS and tokens[i+1]["pos"] in NOUN_POS and is_neg(tokens[i+2]) and tokens[i+3]["pos"] in ADJ_POS):
            add([tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]], "R9_NOUN+NOUN+NEG+ADJ")
    for i in range(n):
        if tokens[i]["pos"] in ADJ_POS:
            head_id = tokens[i]["head_id"]
            if 0 <= head_id < n and tokens[head_id]["pos"] in NOUN_POS:
                add([tokens[head_id], tokens[i]], "R10_DEP_ADJ")
    return phrases

@st.cache_resource(show_spinner=False)
def load_stanza_pipeline():
    import stanza
    stanza.download("id", verbose=False)
    return stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma,depparse", verbose=False, use_gpu=False)

@st.cache_resource(show_spinner=False)
def load_nb_model(repo_id, token):
    from huggingface_hub import hf_hub_download
    model     = joblib.load(hf_hub_download(repo_id=repo_id, filename="naive_bayes_model.joblib",      token=token))
    vectorizer = joblib.load(hf_hub_download(repo_id=repo_id, filename="tfidf_naive_bayes_vectorizer.joblib", token=token))
    with open(hf_hub_download(repo_id=repo_id, filename="naive_bayes_lexicon.json", token=token), "r", encoding="utf-8") as f:
        lexicon = set(json.load(f))
    return model, vectorizer, lexicon

@st.cache_resource(show_spinner=False)
def load_svm_model(repo_id, token):
    from huggingface_hub import hf_hub_download
    model      = joblib.load(hf_hub_download(repo_id=repo_id, filename="svm_model.joblib",           token=token))
    vectorizer = joblib.load(hf_hub_download(repo_id=repo_id, filename="tfidf_svm_vectorizer.joblib", token=token))
    with open(hf_hub_download(repo_id=repo_id, filename="svm_lexicon.json", token=token), "r", encoding="utf-8") as f:
        lex_data = json.load(f)
    lexicon = set(lex_data.get("positive_tokens", {}).keys()) | set(lex_data.get("negative_tokens", {}).keys())
    return model, vectorizer, lexicon

@st.cache_resource(show_spinner=False)
def load_lstm_model(repo_id, token):
    import torch
    import torch.nn as nn
    import compress_fasttext
    from huggingface_hub import hf_hub_download

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
            self.fc   = nn.Linear(hidden_dim, num_classes)

        def forward(self, input_ids):
            embedded    = self.embedding(input_ids)
            out, _      = self.lstm(embedded)
            mean_hidden = torch.mean(out, dim=1)
            return self.fc(mean_hidden)

    ft_path      = hf_hub_download(repo_id=repo_id, filename="fasttext_lstm_vectorizer.bin", token=token)
    weights_path = hf_hub_download(repo_id=repo_id, filename="lstm_model.pt",               token=token)
    lexicon_path = hf_hub_download(repo_id=repo_id, filename="lstm_lexicon.json",            token=token)

    ft_model   = compress_fasttext.models.CompressedFastTextKeyedVectors.load(ft_path)

    state_dict = torch.load(weights_path, map_location="cpu")
    vocab_size = state_dict["embedding.weight"].shape[0]
    model      = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(state_dict)
    model.eval()

    with open(lexicon_path, "r", encoding="utf-8") as f:
        lexicon = set(json.load(f))
    return model, ft_model, lexicon

@st.cache_resource(show_spinner=False)
def load_indobert_model(repo_id, token):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import hf_hub_download
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    model     = AutoModelForSequenceClassification.from_pretrained(repo_id, token=token)
    model.eval()
    with open(hf_hub_download(repo_id=repo_id, filename="indobert_lexicon.json", token=token), "r", encoding="utf-8") as f:
        lexicon = set(json.load(f))
    return model, tokenizer, lexicon

def filter_by_lexicon(phrase_list, lexicon):
    if not lexicon:
        return phrase_list
    return [p for p in phrase_list if any(w in lexicon for w in p["phrase"].lower().split())]

def lemmatize_phrase_for_nb(phrase, nlp_stanza):
    try:
        doc = nlp_stanza(phrase)
        tokens = []
        for sent in doc.sentences:
            for word in sent.words:
                lemma = (word.lemma.lower() if word.lemma else word.text.lower())
                if lemma and len(lemma) >= 2 and lemma not in STOPWORDS:
                    tokens.append(lemma)
        return " ".join(tokens) if tokens else phrase
    except Exception:
        return phrase

def _get_phrase_embeddings_tfidf(phrases, vectorizer):
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    vecs = vectorizer.transform(phrases)
    return vecs.toarray()

def _get_phrase_embeddings_ft(phrases, ft_model):
    phrase_vectors = []
    for phrase in phrases:
        words = phrase.split()
        vectors = []
        for w in words:
            try:
                vectors.append(ft_model[w])
            except KeyError:
                pass
        phrase_vectors.append(np.mean(vectors, axis=0) if vectors else np.zeros(EMBEDDING_DIM))
    return np.array(phrase_vectors)

def _get_phrase_embeddings_indobert(phrases, model, tokenizer):
    import torch
    embeddings = []
    for i in range(0, len(phrases), 64):
        batch = phrases[i:i + 64]
        encodings = tokenizer(batch, truncation=True, padding=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

def semantic_deduplication(phrase_counts, phrase_sentiments, embed_fn, threshold=SIMILARITY_THRESHOLD):
    from sklearn.metrics.pairwise import cosine_similarity

    def cluster(phrases_subset):
        if not phrases_subset:
            return []
        if len(phrases_subset) == 1:
            p = phrases_subset[0]
            return [(p, phrase_counts[p])]
        try:
            embeddings = embed_fn(phrases_subset)
            sim_matrix = cosine_similarity(embeddings)
        except Exception:
            return [(p, phrase_counts[p]) for p in phrases_subset]
        merged = [False] * len(phrases_subset)
        result = []
        for i, phrase in enumerate(phrases_subset):
            if merged[i]:
                continue
            total_count = phrase_counts[phrase]
            for j in range(i + 1, len(phrases_subset)):
                if not merged[j] and sim_matrix[i][j] >= threshold:
                    total_count += phrase_counts[phrases_subset[j]]
                    merged[j] = True
            result.append((phrase, total_count))
        return result

    pos_phrases_list = [p for p, s in phrase_sentiments.items() if s == "positif"]
    neg_phrases_list = [p for p, s in phrase_sentiments.items() if s == "negatif"]

    pos_clustered = cluster(pos_phrases_list)
    neg_clustered = cluster(neg_phrases_list)

    pos_clustered.sort(key=lambda x: x[1], reverse=True)
    neg_clustered.sort(key=lambda x: x[1], reverse=True)
    return pos_clustered, neg_clustered

def predict_nb(phrase, model, vectorizer, lexicon, threshold_pos=0.75, threshold_neg=0.75, nlp_stanza=None):
                                                                                                     
    nb_input = lemmatize_phrase_for_nb(phrase, nlp_stanza) if nlp_stanza is not None else phrase
    vec     = vectorizer.transform([nb_input])
    proba   = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    pi      = next((i for i, c in enumerate(classes) if c in (1, "1", "positif")), 1)
    ni      = 1 - pi
    pos, neg = float(proba[pi]), float(proba[ni])
    if pos >= threshold_pos:
        return "positif", pos, pos, neg
    if neg >= threshold_neg:
        return "negatif", neg, pos, neg
    label = "positif" if pos >= neg else "negatif"
    return label, max(pos, neg), pos, neg

def predict_svm(phrase, model, vectorizer, lexicon, threshold=0.75):
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase.strip():
        normalized_phrase = phrase
    vec      = vectorizer.transform([normalized_phrase])
    decision = float(model.decision_function(vec)[0])
    pos      = 1.0 / (1.0 + np.exp(-decision))
    neg      = 1.0 - pos
    label    = "positif" if pos >= neg else "negatif"
    return label, max(pos, neg), pos, neg

def _tokenize_for_lstm(text, ft_model):
    tokens = text.split()
    vecs = []
    for t in tokens:
        try:
            vecs.append(ft_model[t])
        except KeyError:
            vecs.append(np.zeros(EMBEDDING_DIM))
    return vecs if vecs else [np.zeros(EMBEDDING_DIM)]

def predict_lstm_phrase(phrase, model, ft_model, threshold=0.50):
    import torch
    import torch.nn.functional as F
    vecs = _tokenize_for_lstm(phrase, ft_model)
    x    = torch.tensor(np.array([vecs]), dtype=torch.float32)
    with torch.no_grad():
        out, _      = model.lstm(x)
        mean_hidden = torch.mean(out, dim=1)
        logits      = model.fc(mean_hidden)
        proba       = F.softmax(logits, dim=1)[0].numpy()
    pos, neg = float(proba[1]), float(proba[0])
    label = "positif" if pos >= threshold else "negatif"
    return label, pos if label == "positif" else neg, pos, neg

def predict_indobert(phrase, model, tokenizer, lexicon, threshold=0.75):
    import torch
    import torch.nn.functional as F
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        proba = F.softmax(model(**inputs).logits, dim=1)[0].numpy()
    id2label = model.config.id2label
    pi   = next((k for k, v in id2label.items() if "pos" in str(v).lower()), 1)
    ni   = 1 - pi
    pos, neg = float(proba[pi]), float(proba[ni])
    label = "positif" if pos >= threshold else "negatif"
    return label, pos if label == "positif" else neg, pos, neg

def run_all_models(text, token):
    nlp = load_stanza_pipeline()
    normalized = normalize_text(text)
    clauses    = split_into_clauses(normalized) or [normalized]
    all_phrases_raw = []
    for clause in clauses:
        all_phrases_raw.extend(extract_phrases(clause, nlp))

    results = []
    for model_name in ["Naive Bayes", "SVM", "LSTM", "IndoBERT"]:
        cfg = MODELS_CONFIG[model_name]
        try:
            if cfg["type"] == "nb":
                m, v, l = load_nb_model(cfg["repo"], token=token)
                phrases  = filter_by_lexicon(all_phrases_raw, l)
                if not phrases:
                    phrases = [{"phrase": normalized, "pattern": "fallback"}]
                preds = [predict_nb(p["phrase"], m, v, l, cfg["threshold_pos"], cfg["threshold_neg"], nlp_stanza=nlp) for p in phrases]
                pos_scores = [p[2] for p in preds]
                neg_scores = [p[3] for p in preds]
                avg_pos    = float(np.mean(pos_scores))
                avg_neg    = float(np.mean(neg_scores))
                label      = "positif" if avg_pos >= avg_neg else "negatif"
                conf       = avg_pos if label == "positif" else avg_neg
            elif cfg["type"] == "svm":
                m, v, l = load_svm_model(cfg["repo"], token=token)
                phrases  = filter_by_lexicon(all_phrases_raw, l)
                if not phrases:
                    phrases = [{"phrase": normalized, "pattern": "fallback"}]
                preds = [predict_svm(p["phrase"], m, v, l, cfg["threshold"]) for p in phrases]
                pos_scores = [p[2] for p in preds]
                neg_scores = [p[3] for p in preds]
                avg_pos    = float(np.mean(pos_scores))
                avg_neg    = float(np.mean(neg_scores))
                label      = "positif" if avg_pos >= avg_neg else "negatif"
                conf       = avg_pos if label == "positif" else avg_neg
            elif cfg["type"] == "lstm":
                m, ft, l = load_lstm_model(cfg["repo"], token=token)
                phrases   = filter_by_lexicon(all_phrases_raw, l)
                if not phrases:
                    phrases = [{"phrase": normalized, "pattern": "fallback"}]
                preds = [predict_lstm_phrase(p["phrase"], m, ft, cfg["threshold"]) for p in phrases]
                pos_scores = [p[2] for p in preds]
                neg_scores = [p[3] for p in preds]
                avg_pos    = float(np.mean(pos_scores))
                avg_neg    = float(np.mean(neg_scores))
                label      = "positif" if avg_pos >= cfg["threshold"] else "negatif"
                conf       = avg_pos if label == "positif" else avg_neg
            else:
                m, tok, l = load_indobert_model(cfg["repo"], token=token)
                phrases    = filter_by_lexicon(all_phrases_raw, l)
                if not phrases:
                    phrases = [{"phrase": normalized, "pattern": "fallback"}]
                preds = [predict_indobert(p["phrase"], m, tok, l, cfg["threshold"]) for p in phrases]
                pos_scores = [p[2] for p in preds]
                neg_scores = [p[3] for p in preds]
                avg_pos    = float(np.mean(pos_scores))
                avg_neg    = float(np.mean(neg_scores))
                label      = "positif" if avg_pos >= cfg["threshold"] else "negatif"
                conf       = avg_pos if label == "positif" else avg_neg
            results.append({"model": model_name, "sentiment": label,
                            "confidence": conf, "pos_score": avg_pos, "neg_score": avg_neg, "error": None})
        except Exception as e:
            results.append({"model": model_name, "sentiment": "error",
                            "confidence": 0.0, "pos_score": 0.0, "neg_score": 0.0, "error": str(e)})
    return results

def _predict_sentence(normalized, cfg, m, v_or_ft, l, nlp, tok=None):
    if cfg["type"] == "nb":
        label, *_ = predict_nb(normalized, m, v_or_ft, l, cfg["threshold_pos"], cfg["threshold_neg"], nlp_stanza=nlp)
    elif cfg["type"] == "svm":
        label, *_ = predict_svm(normalized, m, v_or_ft, l, cfg["threshold"])
    elif cfg["type"] == "lstm":
        label, *_ = predict_lstm_phrase(normalized, m, v_or_ft, cfg["threshold"])
    else:
        label, *_ = predict_indobert(normalized, m, tok, l, cfg["threshold"])
    return label

def run_single_model_batch(reviews, model_name, token):
    cfg = MODELS_CONFIG[model_name]
    try:
        if cfg["type"] == "nb":
            m, v, l = load_nb_model(cfg["repo"], token=token)
            tok = None
        elif cfg["type"] == "svm":
            m, v, l = load_svm_model(cfg["repo"], token=token)
            tok = None
        elif cfg["type"] == "lstm":
            m, v, l = load_lstm_model(cfg["repo"], token=token)
            tok = None
        else:
            m, tok, l = load_indobert_model(cfg["repo"], token=token)
            v = None
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return [], [], 0, 0

    nlp = load_stanza_pipeline()
    phrase_counts, phrase_sentiments = {}, {}
    sentence_pos = 0
    sentence_neg = 0
    bar   = st.progress(0, text="Analyzing reviews...")
    total = len(reviews)

    for i, review in enumerate(reviews):
        try:
            normalized = normalize_text(str(review))
            if not normalized.strip():
                bar.progress((i + 1) / total, text=f"Analyzing reviews... ({i+1}/{total})")
                continue

            try:
                sent_label = _predict_sentence(normalized, cfg, m, v, l, nlp, tok)
                if sent_label == "positif":
                    sentence_pos += 1
                else:
                    sentence_neg += 1
            except Exception:
                pass

            clauses = split_into_clauses(normalized) or [normalized]
            for clause in clauses:
                raw_phrases = extract_phrases(clause, nlp)
                filtered    = filter_by_lexicon(raw_phrases, l)
                for p in filtered:
                    phrase = p["phrase"]
                    try:
                        if cfg["type"] == "nb":
                            label, *_ = predict_nb(phrase, m, v, l, cfg["threshold_pos"], cfg["threshold_neg"], nlp_stanza=nlp)
                        elif cfg["type"] == "svm":
                            label, *_ = predict_svm(phrase, m, v, l, cfg["threshold"])
                        elif cfg["type"] == "lstm":
                            label, *_ = predict_lstm_phrase(phrase, m, v, cfg["threshold"])
                        else:
                            label, *_ = predict_indobert(phrase, m, tok, l, cfg["threshold"])
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                        phrase_sentiments.setdefault(phrase, label)
                    except Exception:
                        continue
        except Exception:
            pass
        bar.progress((i + 1) / total, text=f"Analyzing reviews... ({i+1}/{total})")

    bar.empty()

    if phrase_counts:
        if cfg["type"] in ("nb", "svm"):
            embed_fn = lambda phrases: _get_phrase_embeddings_tfidf(phrases, v)
        elif cfg["type"] == "lstm":
            embed_fn = lambda phrases: _get_phrase_embeddings_ft(phrases, v)
        else:
            embed_fn = lambda phrases: _get_phrase_embeddings_indobert(phrases, m, tok)

        pos_phrases, neg_phrases = semantic_deduplication(phrase_counts, phrase_sentiments, embed_fn)
    else:
        pos_phrases, neg_phrases = [], []

    return pos_phrases, neg_phrases, sentence_pos, sentence_neg

tab_single, tab_file = st.tabs(["✏️  Input Single Review", "📂  File Upload"])

with tab_single:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.92rem; color:var(--text-muted); margin-bottom:1rem;">Enter an Indonesian product review sentence. All four models will analyze it simultaneously and return sentiment predictions with confidence scores.</p>', unsafe_allow_html=True)

    review_input    = st.text_area("Review sentence", placeholder="Contoh: Barang datang cepat, kualitasnya bagus banget, tapi packagingnya sedikit lecek.", height=110, label_visibility="collapsed")
    analyze_clicked = st.button("Analyze", key="analyze_single")

    if analyze_clicked:
        if not review_input.strip():
            st.warning("Please enter a review sentence before analyzing.")
        else:
            with st.spinner("Loading models and analyzing..."):
                results = run_all_models(review_input, HF_TOKEN)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="display:flex; align-items:center; gap:8px; margin-bottom:1.25rem;"><span style="font-family:\'Sora\',sans-serif; font-weight:700; font-size:1rem; color:var(--text-main);">Analysis Results</span><div style="flex:1; height:1px; background:var(--border);"></div><span style="font-size:0.78rem; color:var(--text-muted); font-weight:500;">4 models</span></div>', unsafe_allow_html=True)
            cols = st.columns(2, gap="small")
            for idx, r in enumerate(results):
                with cols[idx % 2]:
                    if r["error"]:
                        st.markdown(f'<div class="result-card"><div class="model-tag">{r["model"]}</div><div style="color:var(--negative); font-size:0.88rem;">Error: {r["error"]}</div></div>', unsafe_allow_html=True)
                    else:
                        sc  = "sentiment-pos" if r["sentiment"] == "positif" else "sentiment-neg"
                        em  = "✅" if r["sentiment"] == "positif" else "❌"
                        bc  = "bar-pos" if r["sentiment"] == "positif" else "bar-neg"
                        pct = r["confidence"] * 100
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="model-tag">{r["model"]}</div>
                            <div class="sentiment-label {sc}">{em} {r["sentiment"].capitalize()}</div>
                            <div style="display:flex; justify-content:space-between; margin-top:0.75rem; font-size:0.82rem; color:var(--text-muted); font-weight:500;">
                                <span>Confidence</span><span style="color:var(--text-main); font-weight:700;">{r["confidence"]:.0%}</span>
                            </div>
                            <div class="confidence-bar-wrap"><div class="confidence-bar {bc}" style="width:{pct}%"></div></div>
                            <div style="display:flex; justify-content:space-between; margin-top:0.75rem; font-size:0.82rem;">
                                <span style="color:var(--positive); font-weight:600;">Positif: {r["pos_score"]:.0%}</span>
                                <span style="color:var(--negative); font-weight:600;">Negatif: {r["neg_score"]:.0%}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

with tab_file:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.92rem; color:var(--text-muted); margin-bottom:1.25rem;">Upload a file containing product reviews (one review per row/line). Select a model and display limit, then click Analyze to see aggregated sentiment results.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload reviews file", type=["csv","xlsx","xls","txt"], label_visibility="collapsed", help="CSV, Excel, or plain text. One review per row/line.")

    all_reviews = []
    if uploaded_file is not None:
        _ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        try:
            if _ext == "txt":
                all_reviews = [l.strip() for l in uploaded_file.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
            elif _ext == "csv":
                _raw = uploaded_file.read().decode("utf-8", errors="ignore")
                uploaded_file.seek(0)
                import csv as _csv
                _first_line = _raw.splitlines()[0] if _raw.strip() else ""
                try:
                    _csv.Sniffer().sniff(_first_line)
                    _has_header = _csv.Sniffer().has_header(_raw[:4096])
                except Exception:
                    _has_header = False
                _df = pd.read_csv(
                    uploaded_file,
                    header=0 if _has_header else None,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                    quoting=_csv.QUOTE_MINIMAL,
                )
                all_reviews = _df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                _df = pd.read_excel(uploaded_file)
                all_reviews = _df[_df.columns[0]].dropna().astype(str).tolist()
        except Exception as _e:
            st.error(f"Could not read file: {_e}")
            st.stop()

    c1, c2 = st.columns([1,1], gap="small")
    with c1:
        selected_model = st.selectbox("Model", options=["Naive Bayes","SVM","LSTM","IndoBERT"], index=3)
    with c2:
        display_limit  = st.selectbox("Display limit", options=["Top 10","Top 20","All"], index=0)

    if all_reviews:
        total_rows = len(all_reviews)
        num_rows = st.slider(
            "Number of rows to analyze",
            min_value=1,
            max_value=total_rows,
            value=total_rows,
            step=1,
            help=f"Slide to choose how many of the {total_rows} rows you want to include in the analysis.",
        )
    else:
        num_rows = None

    analyze_file_clicked = st.button("Analyze", key="analyze_file")

    if analyze_file_clicked:
        if uploaded_file is None:
            st.warning("Please upload a file before analyzing.")
        else:
            if not all_reviews:
                st.warning("No reviews found in the uploaded file.")
                st.stop()

            reviews = all_reviews[:num_rows]

            pos_phrases, neg_phrases, sentence_pos, sentence_neg = run_single_model_batch(reviews, selected_model, HF_TOKEN)

            total_sentences = (sentence_pos + sentence_neg) or 1

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:1.5rem;"><span style="font-family:\'Sora\',sans-serif; font-weight:700; font-size:1rem; color:var(--text-main);">Results - {selected_model}</span><div style="flex:1; height:1px; background:var(--border);"></div><span style="font-size:0.78rem; color:var(--text-muted); font-weight:500;">{len(reviews)} reviews loaded</span></div>', unsafe_allow_html=True)

            o1, o2 = st.columns(2, gap="small")
            with o1:
                st.markdown(f'<div class="overview-stat"><div class="ov-pct" style="color:var(--positive);">{sentence_pos/total_sentences:.0%}</div><div class="ov-label" style="color:var(--positive);">✅ Positif</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{sentence_pos} reviews</div></div>', unsafe_allow_html=True)
            with o2:
                st.markdown(f'<div class="overview-stat"><div class="ov-pct" style="color:var(--negative);">{sentence_neg/total_sentences:.0%}</div><div class="ov-label" style="color:var(--negative);">❌ Negatif</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{sentence_neg} reviews</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            total_all_phrases = sum(c for _, c in pos_phrases) + sum(c for _, c in neg_phrases)
            total_all_phrases = total_all_phrases or 1

            for s_label, phrases, bar_cls, s_color in [
                ("Positif Phrases", pos_phrases, "phrase-bar-pos", "var(--positive)"),
                ("Negatif Phrases", neg_phrases, "phrase-bar-neg", "var(--negative)"),
            ]:
                shown = phrases[:10] if display_limit == "Top 10" else phrases[:20] if display_limit == "Top 20" else phrases
                if not shown:
                    continue
                max_count = shown[0][1]
                st.markdown(f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:1rem; margin-top:1.5rem;"><span style="font-family:\'Sora\',sans-serif; font-weight:700; font-size:0.95rem; color:{s_color};">{s_label}</span><span style="font-size:0.78rem; color:var(--text-muted); font-weight:500; background:var(--border); border-radius:99px; padding:2px 9px;">{len(shown)} shown</span><div style="flex:1; height:1px; background:var(--border);"></div></div>', unsafe_allow_html=True)
                st.markdown('<div style="display:flex; align-items:center; gap:1rem; padding:0.5rem 0 0.75rem; border-bottom:2px solid var(--border); margin-bottom:0.25rem;"><span style="min-width:180px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:var(--text-light);">Phrase</span><span style="flex:1; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:var(--text-light);">Frequency</span><span style="min-width:80px; text-align:right; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:var(--text-light);">Count · %</span></div>', unsafe_allow_html=True)
                for phrase_text, count in shown:
                    bw  = (count / max_count) * 100
                    pct = count / total_all_phrases
                    st.markdown(f'<div class="phrase-row"><span class="phrase-text">{phrase_text}</span><div class="phrase-bar-wrap"><div class="{bar_cls}" style="width:{bw}%"></div></div><span class="phrase-count">{count} · {pct:.0%}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="display:flex; align-items:center; gap:8px; margin-bottom:1.25rem; margin-top:1rem;"><span style="font-family:\'Sora\',sans-serif; font-weight:700; font-size:1rem; color:var(--text-main);">Phrase Summary</span><div style="flex:1; height:1px; background:var(--border);"></div></div>', unsafe_allow_html=True)

            total_pos_phrase_count = sum(c for _, c in pos_phrases)
            total_neg_phrase_count = sum(c for _, c in neg_phrases)
            total_phrase_count     = (total_pos_phrase_count + total_neg_phrase_count) or 1
            pos_phrase_pct = total_pos_phrase_count / total_phrase_count
            neg_phrase_pct = total_neg_phrase_count / total_phrase_count
            top_pos = pos_phrases[0][0] if pos_phrases else "-"
            top_neg = neg_phrases[0][0] if neg_phrases else "-"

            box_style = "height: 100%; min-height: 155px; display: flex; flex-direction: column; justify-content: space-between;"
            ps1, ps2, ps3, ps4 = st.columns(4, gap="small")
            with ps1:
                st.markdown(f'<div class="overview-stat" style="{box_style}"><div class="ov-pct" style="color:var(--positive);">{pos_phrase_pct:.0%}</div><div class="ov-label" style="color:var(--positive);">Positive Phrases</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{total_pos_phrase_count} occurrences</div></div>', unsafe_allow_html=True)
            with ps2:
                st.markdown(f'<div class="overview-stat" style="{box_style}"><div class="ov-pct" style="color:var(--negative);">{neg_phrase_pct:.0%}</div><div class="ov-label" style="color:var(--negative);">Negative Phrases</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{total_neg_phrase_count} occurrences</div></div>', unsafe_allow_html=True)
            with ps3:
                st.markdown(f'<div class="overview-stat" style="{box_style}"><div style="font-size:1rem; font-weight:700; color:var(--positive); margin-bottom:0.35rem;">&#127942; {top_pos}</div><div class="ov-label" style="color:var(--text-muted);">Top Positive Phrase</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{pos_phrases[0][1] if pos_phrases else 0}&#215; mentioned</div></div>', unsafe_allow_html=True)
            with ps4:
                st.markdown(f'<div class="overview-stat" style="{box_style}"><div style="font-size:1rem; font-weight:700; color:var(--negative); margin-bottom:0.35rem;">&#9888;&#65039; {top_neg}</div><div class="ov-label" style="color:var(--text-muted);">Top Negative Phrase</div><div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">{neg_phrases[0][1] if neg_phrases else 0}&#215; mentioned</div></div>', unsafe_allow_html=True)
