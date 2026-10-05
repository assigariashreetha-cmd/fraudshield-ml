
import pandas as pd

df = pd.read_csv(r"C:\Users\PC\Downloads\Phishing_URLDataset.csv")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
if "label" in df.columns:
    print(df["label"].value_counts())
else:
    print("Label column not found.")