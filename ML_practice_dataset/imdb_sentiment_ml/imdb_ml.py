import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow_datasets as tfds

# Set random seed for reproducibility
np.random.seed(42)


(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews/plain_text',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)
# Load the IMDB dataset 
data = fetch_openml('IMDB', version=1, as_frame=True)
X = data.data['text']  # Text reviews
y = data.target  # Labels: 0 (negative), 1 (positive)

# Create a pandas DataFrame for the dataset (first 100 samples for display)_
df = pd.DataFrame({'text': X[:100], 'sentiment': y[:100]})
print("\nIMDB Dataset (first 5 rows, first 100 samples):")
print(df.head())

# Display Summary Statistics
print("\nDataset Summatry Statistics:")
print(df.describe())
print("\nSentiment Disribution")
print(df['sentiment'].value_counts())

# Visualize sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution (First 100 Samples)')
plt.xlabel('Sentiment (0=Negative, 1=Positive)')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.show()

# Verify dataset size
print(f"\nTotal samples: {len(X)}")

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Split data into training (80%) and testing (20%) sets)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# Calculate predictions and performance metrics
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"\nTraining accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')
print(f"\nCross-validation accuract scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2f}(+- {cv_scores.std() * 2:.2f})")

# Visualize results: Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report (Test Set)")
print(classification_report(y_test, test_pred, target_names=['Negative', "Positive"]))

# Example: Predict sentment for a test sample
sample_index = 0
new_review_vector = X_test[sample_index]
prediction = model.predict(new_review_vector)
print(f'\nSample review (truncated): {X.iloc[y_test.index[sample_index]][:100]}..')

