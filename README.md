# 🎬 Movie Genre Classification using Machine Learning

## 📌 Objective

The goal of this project is to build a machine learning model that can classify movies into genres based on their plot descriptions. This project was completed as part of a Machine Learning internship task.

---

## 📁 Dataset Overview

The dataset used consists of movie metadata and descriptions split into the following files:

- `train_data.txt`: Contains movie ID, title, genre, and description. Used to train the model.
- `test_data.txt`: Contains movie ID, title, and description. Used to predict genres.
- `test_data_solution.txt`: Contains movie title, genre, and description. Used to evaluate the model's performance.

All fields are separated by ` ::: `.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK (for text preprocessing)
- TF-IDF (Text Vectorization)
- Logistic Regression (Classifier)

---

## 📊 Features & Highlights

- Cleaned and normalized movie descriptions using NLP
- Converted text into numerical vectors using TF-IDF
- Built a Logistic Regression classifier to predict genres
- Matched and evaluated predictions against test labels
- Handled memory-safe evaluation using smart label filtering

---

## 🚀 How to Run the Project

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
