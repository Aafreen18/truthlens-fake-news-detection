import sys
import os
import re
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "fake_news_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "vectorizer.pkl")
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    try:
        if len(sys.argv) < 2:
            print("error")
            return

        text = clean_text(" ".join(sys.argv[1:]))

        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)[0]

        label = "real" if prediction == 1 else "fake"

        print(label)  # ← was missing

    except Exception:
        print("error")


if __name__ == "__main__":
    main()