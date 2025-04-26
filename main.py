# 📦 Import libraries
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 📥 Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# 📂 File paths
train_path = r"C:/Users/MEGHANA/Downloads/Genre Classification Dataset/train_data.txt"
test_path = r"C:/Users/MEGHANA/Downloads/Genre Classification Dataset/test_data.txt"
solution_path = r"C:/Users/MEGHANA/Downloads/Genre Classification Dataset/test_data_solution.txt"

# ✅ Step 1: Check if required files exist
required_files = [train_path, test_path, solution_path]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Missing file: {file}. Please double-check the file paths!")

# 🧼 Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stop_words)

# 📚 Load dataset functions
def load_train():
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split(" ::: ") for line in f if len(line.strip().split(" ::: ")) == 4]
    df = pd.DataFrame(lines, columns=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df["TITLE"] = df["TITLE"].str.strip()
    df["GENRE"] = df["GENRE"].str.lower().str.strip()
    return df

def load_test():
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split(" ::: ") for line in f if len(line.strip().split(" ::: ")) == 3]
    df = pd.DataFrame(lines, columns=["ID", "TITLE", "DESCRIPTION"])
    df["TITLE"] = df["TITLE"].str.strip()
    return df

def load_solution():
    df = pd.read_csv(solution_path, sep=r"\s*:::\s*", engine="python", header=None, names=["TITLE", "GENRE", "DESCRIPTION"])
    df["TITLE"] = df["TITLE"].str.strip()
    df["GENRE"] = df["GENRE"].str.lower().str.strip()
    return df

# 📥 Load datasets
print("📥 Loading datasets...")
train_df = load_train()
test_df = load_test()
solution_df = load_solution()

# 🧽 Clean descriptions
print("🧼 Cleaning descriptions...")
train_df["CLEAN_DESC"] = train_df["DESCRIPTION"].apply(clean_text)
test_df["CLEAN_DESC"] = test_df["DESCRIPTION"].apply(clean_text)

# 🔢 TF-IDF vectorization
print("🔢 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df["CLEAN_DESC"])
X_test = vectorizer.transform(test_df["CLEAN_DESC"])

# 🎯 Label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["GENRE"])

# 🤖 Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20),
    "Linear SVM": LinearSVC(max_iter=1000)
}

# 🧪 Train and evaluate
results = []

print("\n🔁 Comparing models...\n")
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_df["PREDICTED_GENRE"] = y_pred
    merged_df = pd.merge(test_df[["TITLE", "PREDICTED_GENRE"]],
                         solution_df[["TITLE", "GENRE"]],
                         on="TITLE", how="inner")

    merged_df["GENRE"] = merged_df["GENRE"].str.lower().str.strip()
    merged_df["PREDICTED_GENRE"] = merged_df["PREDICTED_GENRE"].str.lower().str.strip()

    acc = accuracy_score(merged_df["GENRE"], merged_df["PREDICTED_GENRE"])
    print(f"✅ {name} Accuracy: {acc:.4f}")
    results.append((name, acc))

    # 📈 Print classification report
    print("\n📈 Classification Report:")
    print(classification_report(merged_df["GENRE"], merged_df["PREDICTED_GENRE"]))

# 🏁 Print final model comparison
print("\n🏁 Final Model Comparison:")
for name, acc in results:
    print(f"{name:<20}: {acc:.4f}")

best_model = max(results, key=lambda x: x[1])
print(f"\n🥇 Best Model: {best_model[0]} with Accuracy: {best_model[1]:.4f}")
