# TruthLens: Fake News Detection Using NLP and Machine Learning

## Overview

TruthLens is a machine learning-based fake news detection system that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques. The system analyzes the textual content of news articles and predicts their authenticity based on patterns learned from large-scale news datasets.

The project combines a Python-based machine learning model, a Node.js backend API, and a web-based frontend interface, providing an end-to-end solution for fake news classification.

---

## Live Demo

### Frontend

https://truthlens-fakenews-detector.netlify.app/

### Backend API

https://truthlens-fake-news-detection.onrender.com/

---

## Features

* Detects whether a news article is Real or Fake
* User-friendly web interface
* Machine Learning-powered predictions
* NLP-based text preprocessing
* REST API backend using Express.js
* Python model integration using Python Shell
* Deployable on Render and Netlify
* Real-time prediction results

---

## Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Node.js
* Express.js
* Python Shell
* CORS

### Machine Learning

* Python
* Scikit-learn
* TF-IDF Vectorization
* Linear Support Vector Machine (Linear SVM)
* Joblib

---

## Skills Demonstrated

* Machine Learning
* Natural Language Processing (NLP)
* Text Classification
* Scikit-learn
* Model Deployment
* REST API Development
* Node.js & Express.js
* Frontend Development
* Git & GitHub

---

## Dataset

The model was trained using a combination of:

### FakeNewsNet

A benchmark dataset containing real and fake news articles collected from verified fact-checking sources and social media platforms.

### ISOT Fake News Dataset

A widely used dataset containing:

* Real news articles
* Fake news articles
* Political and current affairs content

### Dataset Note

The datasets are not included in this repository because of GitHub file size limitations.

Users can download the datasets separately and place them inside the `dataset/` directory.

---

## Machine Learning Pipeline

### Data Preprocessing

* Lowercasing
* Removal of URLs
* Removal of punctuation
* Removal of special characters
* Text normalization

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert textual data into numerical vectors.

### Classification Model

A Linear Support Vector Machine (Linear SVM) classifier is trained on the processed news articles.

Pipeline:

```text
News Article
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Linear SVM
      ↓
Prediction
```

---

## Why Linear SVM?

Linear SVM was selected because:

* Performs well on high-dimensional sparse text data.
* Works efficiently with TF-IDF features.
* Faster training and inference compared to many deep learning models.
* Provides strong baseline performance for fake news classification tasks.

---

## Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1-Score | 100.00% |

*Note: These results were obtained on the selected test split of the combined FakeNewsNet and ISOT datasets. Performance on unseen real-world news data may vary.*

---

## Project Architecture

```text
Frontend
   ↓
Express API
   ↓
Python Prediction Service
   ↓
TF-IDF Vectorizer
   ↓
Linear SVM Model
   ↓
Prediction Result
```

---

## Project Structure

```text
FNewsPredictor/
│
├── backend/
│   ├── routes/
│   ├── services/
│   ├── server.js
│   └── package.json
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── ml-model/
│   ├── train_model.py
│   ├── predict.py
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── dataset/
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Aafreen18/truthlens-fake-news-detection.git
cd truthlens-fake-news-detection
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Backend Dependencies

```bash
cd backend
npm install
```

---

## Training the Model

Place the dataset files inside the `dataset/` directory.

Run:

```bash
cd ml-model
python train_model.py
```

This generates:

```text
fake_news_model.pkl
vectorizer.pkl
```

---

## Running the Backend

```bash
cd backend
npm start
```

Server starts on:

```text
http://localhost:3001
```

---

## Running the Frontend

Open:

```text
frontend/index.html
```

or use:

```bash
cd frontend
python3 -m http.server 5500
```

Then visit:

```text
http://localhost:5500
```

---

## API Endpoint

### Predict News Authenticity

**POST**

```http
/predict
```

### Request Body

```json
{
  "title": "News Title",
  "text": "News Article Content"
}
```

### Response

```json
{
  "prediction": "real"
}
```

or

```json
{
  "prediction": "fake"
}
```

---

## Deployment

### Frontend (Netlify)

https://truthlens-fakenews-detector.netlify.app/

### Backend (Render)

https://truthlens-fake-news-detection.onrender.com/

---

## Limitations

* The model does not perform real-time fact-checking.
* Predictions are based on textual patterns learned during training.
* Performance may decrease on domains that are underrepresented in the training data, such as:

  * Scientific news
  * Medical research
  * Sports news
  * Highly technical articles

The model is primarily optimized for political, current affairs, and general news content similar to the training datasets.

---

## Future Improvements

* Deep Learning-based models (LSTM, BERT, RoBERTa)
* Real-time fact-checking integration
* Explainable AI predictions
* News source credibility analysis
* Multi-language fake news detection
* Browser extension support

---

## Author

**Aafreen Parveen**

B.Tech, Electronics and Communication Engineering
Netaji Subhas University of Technology (NSUT), Delhi

---

## License

This project is developed for educational and research purposes.
