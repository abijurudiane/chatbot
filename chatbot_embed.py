import pandas as pd
import numpy as np
import string
import re
import gensim.downloader as api
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. Load data ----------
df = pd.read_csv("data/customer_support_faq_dataset.csv", encoding="utf-8")
df.columns = df.columns.str.strip().str.lower()

# ---------- 2. Cleaning (keep informative words) ----------
# Only remove truly useless tokens. Do NOT remove "not", "no", "can", "how", "why"...
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "on", "in", "at", "by", "with", "to", "from",
    "and", "or", "but", "if", "then", "so",
    "i", "you", "me", "my", "your", "we", "us", "our",
    "this", "that", "these", "those", "it", "its",
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # keep letters/digits
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

df["searchable"] = (
    df["question"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["context"].fillna("")
).apply(clean)

# ---------- 3. Choose ONE embedding backend ----------
BACKEND = "fasttext"   # "word2vec" | "fasttext" | "trained_w2v"

if BACKEND == "word2vec":
    print("Loading Google-News Word2Vec (~1.6 GB, first run)...")
    wv = api.load("word2vec-google-news-300")
    DIM = wv.vector_size
    def word_vec(w): return wv[w] if w in wv else None

elif BACKEND == "fasttext":
    print("Loading FastText wiki-news (~1 GB, first run)...")
    wv = api.load("fasttext-wiki-news-subwords-300")
    DIM = wv.vector_size
    def word_vec(w):
        try:
            return wv[w]          # FastText can produce OOV vectors
        except KeyError:
            return None

elif BACKEND == "trained_w2v":
    print("Training Word2Vec on your FAQ corpus...")
    corpus = [t.split() for t in df["searchable"]]
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,          # skip-gram: better on small data
        epochs=50,
        workers=4,
    )
    wv = model.wv
    DIM = wv.vector_size
    def word_vec(w): return wv[w] if w in wv else None

# ---------- 4. TF-IDF weights (downweight common words) ----------
tfidf = TfidfVectorizer()
tfidf.fit(df["searchable"])
vocab_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
default_weight = np.mean(list(vocab_weights.values()))

def sentence_vector(text):
    tokens = text.split()
    vecs, weights = [], []
    for w in tokens:
        v = word_vec(w)
        if v is not None:
            vecs.append(v)
            weights.append(vocab_weights.get(w, default_weight))
    if not vecs:
        return np.zeros(DIM), 0
    weights = np.array(weights) / np.sum(weights)     # normalize
    return np.average(vecs, axis=0, weights=weights), len(vecs)

# ---------- 5. Precompute FAQ vectors ----------
doc_matrix = np.vstack([sentence_vector(t)[0] for t in df["searchable"]])

# ---------- 6. Keyword overlap ----------
def keyword_overlap(user_clean, doc_clean):
    u, d = set(user_clean.split()), set(doc_clean.split())
    return len(u & d) / len(u) if u else 0.0

# ---------- 7. Match ----------
def get_answer(user_question, threshold=0.35):
    user_clean = clean(user_question)
    if not user_clean:
        return "Please type a question.", 0.0, None

    user_vec, known_count = sentence_vector(user_clean)
    if known_count == 0:
        return (
            "Sorry, I couldn't understand that. Try rephrasing or email support@example.com.",
            0.0, None,
        )

    emb_scores = cosine_similarity(user_vec.reshape(1, -1), doc_matrix)[0]
    overlap_scores = np.array([keyword_overlap(user_clean, d) for d in df["searchable"]])
    combined = 0.8 * emb_scores + 0.2 * overlap_scores

    best_idx = int(np.argmax(combined))
    best_score = float(combined[best_idx])

    if best_score < threshold:
        return (
            "Sorry, I couldn't find a good match. Try rephrasing or email support@example.com.",
            best_score, None,
        )

    row = df.iloc[best_idx]
    return row["answer"], best_score, row["question"]

# ---------- 8. Test loop ----------
if __name__ == "__main__":
    print(f"FAQ Chatbot ready (backend={BACKEND}). Type 'quit' to exit.\n")
    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            break
        answer, score, matched = get_answer(q)
        print(f"Bot (score={score:.2f})")
        if matched:
            print(f"  Matched: {matched}")
        print(f"  Answer : {answer}\n")