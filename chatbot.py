import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. Load data ----------
df = pd.read_csv("data/customer_support_faq_dataset.csv", encoding="utf-8")
df.columns = df.columns.str.strip().str.lower()

STOPWORDS = {
    "how", "do", "i", "the", "a", "an", "is", "are", "can", "to", "my",
    "of", "for", "on", "in", "it", "and", "or", "what", "why", "when",
    "where", "which", "you", "your", "me", "we", "us", "be", "with",
    "this", "that", "there", "was", "were", "will", "would", "should",
    "could", "have", "has", "had", "not", "no", "yes", "if", "at", "by",
}

def clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

# ---------- 3. Build the searchable corpus ----------
# Include question + category + context so matching is richer
df["searchable"] = (
    df["question"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["context"].fillna("")
).apply(clean)

# ---------- 4. TF-IDF ----------
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,   # better weighting for short texts
)
tfidf_matrix = vectorizer.fit_transform(df["searchable"])

# ---------- 5. Keyword overlap helper (extra signal) ----------
def keyword_overlap(user_clean, doc_clean):
    u = set(user_clean.split())
    d = set(doc_clean.split())
    if not u:
        return 0.0
    return len(u & d) / len(u)

# ---------- 6. Match ----------
def get_answer(user_question, threshold=0.05):
    user_clean = clean(user_question)
    if not user_clean:
        return "Please type a question.", 0.0, None

    user_vec = vectorizer.transform([user_clean])
    tfidf_scores = cosine_similarity(user_vec, tfidf_matrix)[0]

    # Combine TF-IDF score with keyword overlap
    combined = []
    for i, s in enumerate(tfidf_scores):
        overlap = keyword_overlap(user_clean, df.iloc[i]["searchable"])
        combined.append(0.7 * s + 0.3 * overlap)

    combined = list(combined)
    best_idx = max(range(len(combined)), key=lambda i: combined[i])
    best_score = combined[best_idx]

    if best_score < threshold:
        return (
            "Sorry, I couldn't find a good match. "
            "Try rephrasing, or contact support@example.com.",
            best_score,
            None,
        )

    row = df.iloc[best_idx]
    return row["answer"], best_score, row["question"]

# ---------- 7. Terminal test ----------
if __name__ == "__main__":
    print("FAQ Chatbot ready. Type 'quit' to exit.\n")
    while True:
        q = input("You: ")
        if q.lower() in ("quit", "exit"):
            break
        answer, score, matched = get_answer(q)
        print(f"Bot (score={score:.2f})")
        if matched:
            print(f"  Matched: {matched}")
        print(f"  Answer : {answer}\n")