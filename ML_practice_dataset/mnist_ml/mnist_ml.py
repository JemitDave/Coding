import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility 
np.random.seed(42)

# Load the MNIST digits dataset (uning load_digits for simplicity)
digits = load_digits()
X = digits.data  # Features: 8x8 pixel values (64 features)
y = digits.target  # Label: 0-9 digits

# Create a pandas DataFrames for the dataset (first 100 samples for display)
df = pd.DataFrame(data=X[:100], columns=[f'pixel_{i}' for i in range (X.shape[1])])
df['digit'] = y[:100]
print("\nMNIST Digits Dataset (first 5 rows, first 100 samples):")
print(df.head())

# Display summary statisctics
print("\nDataset Summary Statistics (first 100 samples):")
print(df.describe())

# Verify Dataset size
print(f'\nTotal Samples: {len(X)}')
print(f'Feature Names: {digits.feature_names[:5]}... (64 pixel total)')
print(f'Target names: {digits.target_names}')


# Visualize sample digits (first 4 images)
plt.figure(figsize=(8,8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(digits.images[i],cmap='gray')
    plt.title(f'Digit: {digits.target[i]}')
    plt.axis('off')
plt.savefig('sample_digits.png')
plt.show()


# Split data into training(80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'\nTraining set size: {len(X_train)} samples')
print(f'Test set size: {len(X_test)} samples')

# Scale the features (SVM performs better with scaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model 
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# Calcuate prefictions and performance metrics
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print(f'\nTraining accuracy: {train_accuracy:.2f}')
print(f'Test accuract: {test_accuracy:.2f}')


# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
print(f'\nCross-validation accuracy scores: {cv_scores}')
print(f'Mean CV accuracyL: {cv_scores.mean():.2f} (+-{cv_scores.std() * 2:2f})')


# Visualize results: Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
plt.figure(figsize=(10,8))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.savefig('confusion_matrix.png')
plt.show()


# Print classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_pred, target_names=[str(i) for i in digits.target_names]))


# Example: Predict digit for a test sample
new_digit = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(new_digit)
plt.figure(figsize=(4, 4))
plt.imshow(X_test[0].reshape(8, 8), cmap='gray')
plt.title(f'Predicted Digit: {prediction[0]}')
plt.axis('off')
plt.savefig('predicted_digit.png')
plt.show()
print(f'\nPredicted digit for test sample: {prediction[0]}')