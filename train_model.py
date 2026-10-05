import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ==========================================
# DATASET
# ==========================================

DATASET_PATH = r"C:\Users\PC\Downloads\Phishing_URLDataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# KEEP ONLY URL + LABEL
# ==========================================

df = df[["URL", "label"]].copy()

df = df.dropna()

df["URL"] = df["URL"].astype(str)

df["label"] = df["label"].astype(int)


print("\nLabel distribution:")
print(df["label"].value_counts())

# Dataset convention:
# 0 = Phishing
# 1 = Legitimate


# ==========================================
# INPUT AND TARGET
# ==========================================

X = df["URL"]

y = df["label"]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# TF-IDF URL FEATURES
# ==========================================

print("\nCreating URL features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=80000,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF features created!")


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\nTraining machine learning model...")

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")


# ==========================================
# TEST MODEL
# ==========================================

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing",
            "Legitimate"
        ]
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

model_data = {

    "model": model,

    "vectorizer": vectorizer

}


with open(
    "phishing_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


print("\n====================================")
print("MODEL SAVED SUCCESSFULLY!")
print("====================================")

print("File: phishing_model.pkl")