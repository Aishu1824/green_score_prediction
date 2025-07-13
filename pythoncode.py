import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ---------- STEP 1: Load CSV ----------
df = pd.read_csv("sustainability_dataset.csv")  # Make sure this file is in the same folder

# ---------- STEP 2: Clean Text ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['description'].fillna("").apply(clean_text)
df['brand'] = df['brand'].fillna("")
df['input_text'] = df['brand'] + " " + df['clean_text']

# ---------- STEP 3: Generate Green Score (if not provided) ----------
keywords = ['biodegradable', 'organic', 'plastic-free', 'eco-friendly', 'compostable', 'recyclable']

def score_keywords(text):
    return sum([text.count(k) for k in keywords])

df['green_score'] = df['clean_text'].apply(score_keywords)

scaler = MinMaxScaler()
df['green_score_scaled'] = scaler.fit_transform(df[['green_score']])

# ---------- STEP 4: TF-IDF + Model ----------
X = df['input_text']
y = df['green_score_scaled']

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = Ridge()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"\n✅ Model trained. Test MSE = {mse:.4f}")

# ---------- STEP 5: Try on New Input ----------
def predict_score(text):
    vec = vectorizer.transform([text])
    return float(model.predict(vec))

sample = "Organic bamboo toothbrush with compostable packaging"
score = predict_score(sample)
print(f"\n🟢 Sample Green Score: {round(score, 2)} (scale: 0–1)")
