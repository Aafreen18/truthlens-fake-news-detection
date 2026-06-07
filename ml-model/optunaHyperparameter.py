import pandas as pd
import re
import optuna

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("../dataset/fakenewsnet_isot_combined.csv")

# ==========================================
# Clean Text
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["text"] = (
    data["article_title"].fillna("")
    + " "
    + data["article_content"].fillna("")
)

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["labels"]

# ==========================================
# Optuna Objective
# ==========================================

def objective(trial):

    vectorizer = TfidfVectorizer(
        stop_words="english",

        ngram_range=trial.suggest_categorical(
            "ngram_range",
            [(1, 1), (1, 2)]
        ),

        min_df=trial.suggest_int(
            "min_df",
            2,
            5
        ),

        max_df=trial.suggest_float(
            "max_df",
            0.85,
            0.95
        ),

        max_features=trial.suggest_categorical(
            "max_features",
            [20000, 30000, 50000]
        ),

        sublinear_tf=trial.suggest_categorical(
            "sublinear_tf",
            [True, False]
        ),

        norm=trial.suggest_categorical(
            "norm",
            ["l2"]
        )
    )

    X_tfidf = vectorizer.fit_transform(X)

    model = LinearSVC(
        C=trial.suggest_float(
            "C",
            0.01,
            10,
            log=True
        ),

        loss=trial.suggest_categorical(
            "loss",
            ["squared_hinge"]
        ),

        class_weight="balanced",

        max_iter=10000,

        random_state=42
    )

    score = cross_val_score(
        model,
        X_tfidf,
        y,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    ).mean()

    return score

# ==========================================
# Run Study
# ==========================================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

# ==========================================
# Results
# ==========================================

print("\nBest Macro F1:")
print(study.best_value)

print("\nBest Parameters:")
print(study.best_params)