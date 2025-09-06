import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Select relevant features and target
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = titanic[features]
y = titanic['survived']

# Create a pandas DataFrame for display
df = pd.DataFrame(X, columns=features)
df['survived'] = y
print("\nTitanic Dataset (first 5 rows):")
print(df.head())

# Display summary statistics
print("\nDataset Summary Statistics:")
print(df.describe(include='all'))

# Verify dataset size
print(f"\nTotal samples: {len(X)}")
print(f"Feature names: {features}")

# Visualize survival rates by passenger class and sex
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df, x='pclass', hue='survived')
plt.title('Survival by Passenger Class')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival by Sex')

plt.tight_layout()
plt.savefig('titanic_features.png')
plt.show()

# Preprocessing: Handle missing values and encode categorical variables
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex']

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            'cat',
            OneHotEncoder(drop='first'),
            categorical_features,
        ),
    ]
)

# Create pipeline with preprocessing and model
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42)),
    ]
)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Train the model
model.fit(X_train, y_train)

# Calculate predictions and performance metrics
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2f} "
      f"(± {cv_scores.std() * 2:.2f})")

# Visualize results: Confusion matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Survived', 'Survived'],
)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.savefig('confusion_matrix.png')
plt.show()

# Visualize results: ROC Curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve')
plt.savefig('roc_curve.png')
plt.show()

# Print classification report
print("\nClassification Report (Test Set):")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=['Not Survived', 'Survived'],
    )
)

# Example: Predict survival for a test sample
new_passenger = X_test.iloc[[0]]
prediction = model.predict(new_passenger)

print(
    f"\nPredicted survival for test sample: "
    f"{'Survived' if prediction[0] == 1 else 'Not Survived'}"
)
print(f"Sample features:\n{df.iloc[X_test.index[0]]}")


# Define new passenger data
new_passengers = pd.DataFrame([
    {
        "pclass": 3,      # 3rd class
        "sex": "male",    # male
        "age": 22,        # 22 years old
        "sibsp": 1,       # 1 sibling/spouse aboard
        "parch": 0,       # 0 parents/children aboard
        "fare": 7.25      # ticket fare
    },
    {
        "pclass": 1,      # 1st class
        "sex": "female",  # female
        "age": 35,        
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0
    },
    {
        "pclass": 2,
        "sex": "female",
        "age": 15,
        "sibsp": 0,
        "parch": 2,
        "fare": 30.0
    }
])


# Predict survival (0 = Not Survived, 1 = Survived)
predictions = model.predict(new_passengers)

# Get prediction probabilities
proba = model.predict_proba(new_passengers)

# Attach predictions to the DataFrame
new_passengers["predicted_survival"] = [
    "Survived" if p == 1 else "Not Survived" for p in predictions
]
new_passengers["prob_not_survived"] = proba[:, 0]
new_passengers["prob_survived"] = proba[:, 1]
print("\nPredictions for new passengers with probabilities:")
print(new_passengers)
