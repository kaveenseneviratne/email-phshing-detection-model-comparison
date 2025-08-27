import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# === Load zero-shot classification pipeline ===
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Define target labels exactly as needed ===
LABELS = ["Safe Email", "Phishing Email"]

# === Load your CSV ===
df = pd.read_csv("test.csv")  

# === Perform classification ===
predictions = []

for email in tqdm(df["Email Text"]):
    if isinstance(email, str) and email.strip():
        result = classifier(email, candidate_labels=LABELS)
        pred_label = result["labels"][0]  # Top prediction
    else:
        pred_label = "unknown"
    predictions.append(pred_label)

# === Store predictions ===
df["Predict"] = predictions
df.to_csv("classified_emails.csv", index=False)
print("✅ Saved classified_emails.csv with predictions.")
