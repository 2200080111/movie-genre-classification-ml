import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Load train data
def load_train():
    path = "C:/Users/MEGHANA/Downloads/Genre Classification Dataset/train_data.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split(" ::: ") for line in f if len(line.strip().split(" ::: ")) == 4]
    df = pd.DataFrame(lines, columns=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df["ID"] = df["ID"].astype(str).str.strip()
    df["TITLE"] = df["TITLE"].astype(str).str.strip()
    df["GENRE"] = df["GENRE"].str.lower().str.strip()  # Normalize genres
    return df

# Load test data
def load_test():
    path = "C:/Users/MEGHANA/Downloads/Genre Classification Dataset/test_data.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split(" ::: ") for line in f if len(line.strip().split(" ::: ")) == 3]
    df = pd.DataFrame(lines, columns=["ID", "TITLE", "DESCRIPTION"])
    df["ID"] = df["ID"].astype(str).str.strip()
    df["TITLE"] = df["TITLE"].astype(str).str.strip()
    return df

#  Load test solution with correct column order
def load_solution():
    path = "C:/Users/MEGHANA/Downloads/Genre Classification Dataset/test_data_solution.txt"
    df = pd.read_csv(path, sep=r"\s*:::\s*", engine="python", header=None, names=["TITLE", "GENRE", "DESCRIPTION"])
    df["TITLE"] = df["TITLE"].astype(str).str.strip()
    df["GENRE"] = df["GENRE"].str.lower().str.strip()  # Normalize genres
    return df

# 📥 Load all datasets
print("📥 Loading datasets...")
train_df = load_train()
test_df = load_test()
solution_df = load_solution()

# 🔍 Debug: Check matching titles
print("🔎 Sample test_df TITLEs:", test_df["TITLE"].head(3).tolist())
print("🔎 Sample solution_df TITLEs:", solution_df["TITLE"].head(3).tolist())
matching_titles = set(test_df["TITLE"]) & set(solution_df["TITLE"])
print(f"🔍 Matching test vs solution TITLEs: {len(matching_titles)}")

# 🧼 Clean descriptions
print("🧼 Cleaning descriptions...")
train_df["CLEAN_DESC"] = train_df["DESCRIPTION"].apply(clean_text)
test_df["CLEAN_DESC"] = test_df["DESCRIPTION"].apply(clean_text)

# 🔢 Vectorize
print("🔢 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["CLEAN_DESC"])
X_test = vectorizer.transform(test_df["CLEAN_DESC"])

# 🎯 Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["GENRE"])

# 🤖 Train model
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔮 Predict on test set
print("🔮 Predicting on test data...")
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
test_df["PREDICTED_GENRE"] = y_pred

# 🧪 Evaluate
print("🧪 Evaluating model...")
merged_df = pd.merge(test_df[["TITLE", "PREDICTED_GENRE"]],
                     solution_df[["TITLE", "GENRE"]],
                     on="TITLE",
                     how="inner")

if merged_df.empty:
    print("❌ Error: No matching TITLES found between test predictions and solution file.")
else:
    print("✅ Evaluation results:")
    accuracy = accuracy_score(merged_df["GENRE"], merged_df["PREDICTED_GENRE"])
    print(f"🎯 Accuracy: {accuracy:.4f}")

    print("🔍 Predicted genres:", merged_df["PREDICTED_GENRE"].unique())
    print("🔍 True genres:", merged_df["GENRE"].unique())

    # Handle matching genres only to avoid memory errors
    common_labels = list(set(merged_df["GENRE"]) & set(merged_df["PREDICTED_GENRE"]))

    if not common_labels:
        print("⚠️ No common genres between predictions and true labels. Cannot compute classification report.")
    else:
        print("\n📈 Classification Report (only for matching genres):\n")
        print(classification_report(
            merged_df["GENRE"],
            merged_df["PREDICTED_GENRE"],
            labels=common_labels
        ))
