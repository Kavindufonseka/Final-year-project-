import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
import time

# Load the dataset
data = pd.read_csv('url.csv')

# Feature extraction function
def extract_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['has_special'] = int(bool(re.search(r'[^a-zA-Z0-9./:]', url)))
    features['domain_length'] = len(urlparse(url).netloc)
    return features

# Apply feature extraction
print("Extracting features...")
start_time = time.time()
features = data['url'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())
print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds.")

# Convert DataFrame to sparse matrix
features_sparse = csr_matrix(features_df.values)

# Define a named function for tokenization
def url_tokenizer(url):
    return re.split(r'\W+', url)

# Combine with TF-IDF features
print("Vectorizing URLs...")
start_time = time.time()
vectorizer = TfidfVectorizer(tokenizer=url_tokenizer, token_pattern=None, max_features=5000)
tfidf_features = vectorizer.fit_transform(data['url'])
print(f"Vectorization completed in {time.time() - start_time:.2f} seconds.")

# Combine all features
X = hstack([tfidf_features, features_sparse])
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with parallel processing
print("Training the model...")
start_time = time.time()
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"Model training completed in {time.time() - start_time:.2f} seconds.")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Cross-validation
print("Performing cross-validation...")
start_time = time.time()
cv_scores = cross_val_score(model, X, y, cv=3)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')
print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds.")

# Save the model and vectorizer
joblib.dump(model, 'optimized_malware_model.pkl')
joblib.dump(vectorizer, 'optimized_vectorizer.pkl')
print("Model and vectorizer saved successfully.")
