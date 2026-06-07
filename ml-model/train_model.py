import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv(
    "../dataset/fakenewsnet_isot_combined.csv",
    encoding="latin1"
)

# ==========================================
# Combine Title + Content
# ==========================================

data["text"] = (
    data["article_title"].fillna("")
    + " "
    + data["article_content"].fillna("")
)

# ==========================================
# Text Cleaning
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["text"] = data["text"].apply(clean_text)

# ==========================================
# Features / Labels
# ==========================================

X = data["text"]
y = data["labels"]

# ==========================================
# Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TF-IDF Vectorizer
# (Optuna Best Parameters)
# ==========================================

vectorizer = TfidfVectorizer(
    norm="l2",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.9105271447996712,
    sublinear_tf=True
)

# ==========================================
# Transform Data
# ==========================================

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# Linear SVM
# (Optuna Best Parameters)
# ==========================================

model = LinearSVC(
    C=0.8022296334440381,
    class_weight="balanced",
    random_state=42,
    max_iter=10000,
    loss="squared_hinge"
)

# ==========================================
# Train
# ==========================================

model.fit(X_train_tfidf, y_train)

# ==========================================
# Predict & Evaluate
# ==========================================

pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")

print("\n==========================")
print("TEST SET RESULTS")
print("==========================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=["Fake (0)", "Real (1)"]))

# ==========================================
# Cross Validation (5-fold, no data leakage)
# ==========================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            max_df=0.9105271447996712,
            sublinear_tf=True
        )
    ),
    (
        "svm",
        LinearSVC(
            C=0.8022296334440381,
            class_weight="balanced",
            random_state=42,
            max_iter=10000,
            loss="squared_hinge"
        )
    )
])

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("\n==========================")
print("CROSS VALIDATION (5-fold)")
print("==========================")
print(f"Macro F1 Scores : {cv_scores}")
print(f"Mean Macro F1   : {cv_scores.mean():.4f}")
print(f"Std             : {cv_scores.std():.4f}")

# ==========================================
# Save Vectorizer & Model Separately
# ==========================================

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "fake_news_model.pkl")

print("\n==========================")
print("MODEL SAVED")
print("==========================")
print("Vectorizer -> vectorizer.pkl")
print("Model      -> fake_news_model.pkl")