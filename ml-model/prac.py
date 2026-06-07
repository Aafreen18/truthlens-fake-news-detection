import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("../dataset/news_dataset.csv", encoding="latin1")

# -------------------------------
# Combine Text Columns
# -------------------------------
df["text"] = (
    df["article_title"].fillna("") + " " +
    df["article_content"].fillna("")
)

# -------------------------------
# Transform Text
# -------------------------------
X = vectorizer.transform(df["text"])

# -------------------------------
# Predictions
# -------------------------------
predictions = model.predict(X)

# -------------------------------
# Actual Labels
# -------------------------------
y_true = df["labels"]

# -------------------------------
# Accuracy
# -------------------------------
accuracy = accuracy_score(y_true, predictions)

# -------------------------------
# F1 Scores
# -------------------------------
f1 = f1_score(y_true, predictions)
macro_f1 = f1_score(y_true, predictions, average="macro")
weighted_f1 = f1_score(y_true, predictions, average="weighted")

print("\n==========================")
print("TEST SET RESULTS")
print("==========================")
print(f"Accuracy     : {accuracy:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Weighted F1  : {weighted_f1:.4f}")

# -------------------------------
# Classification Report
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_true, predictions))

# -------------------------------
# Confusion Matrix
# -------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

# -------------------------------
# Prediction Distribution
# -------------------------------
print("\nPrediction Distribution:")
print(
    pd.Series(predictions)
    .value_counts(normalize=True)
    .sort_index()
)

# -------------------------------
# Number Predicted as Real/Fake
# -------------------------------
fake_count = (predictions == 0).sum()
real_count = (predictions == 1).sum()

print(f"\nPredicted Fake (0): {fake_count}")
print(f"Predicted Real (1): {real_count}")