import pprint
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the California Housing Dataset
housing = fetch_california_housing()
X = housing.data 
# Features: eg., meduian income, house age,etc.
y = housing.target
# target: median house value(in $100,000s)

df = pd.DataFrame(data = X, columns = housing.feature_names)
df['MedHouseVal'] = y
print('\nCalifornia Housing Dataset (first 5 rows:)')
print(df.head())

# Display summary statistics
print("\nDataset Summary Statistics:")
print(df.describe())

# Verify dataset size
print(f"\nTotal samples: {len(X)}")
print(f"Feature names: {housing.feature_names}")

# Visualize the dataset: Scatter plot of median income vs house value
plt.figure(figsize=(8,6))
sns.scatterplot(data = df, x = 'MedInc', y= "MedHouseVal", alpha = 0.5)
plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income ($)")
plt.ylabel('Median House Value ($100,000)')
plt.savefig('housing_scatter.png')
plt.show()

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples ")


# Train the Random Forest model
model = RandomForestRegressor(n_estimators = 200, max_depth=10, random_state = 42)
model.fit(X_train, y_train)


# Calculate predictions and performance metrics
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\nTraining MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")


# Perform 5-fold cross validation
cv_score = cross_val_score(model, X, y, cv = 5, scoring = 'r2')
print(f"\nCross-validation R2 scores: {cv_score}")
print(f"Mean CV R2: {cv_score.mean(): .2f} (+- {cv_score.std() * 2:.2f})")

# Visualize results: Actual vs Predicted house values
plt.figure(figsize = (8,6))
plt.scatter(y_test, test_pred, alpha = 0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)
plt.xlabel('Actual House ($100,000)')
plt.ylabel('Predicted House Value ($100,000)')
plt.title('Actual vs Predicted House Values (Test Set)')
plt.savefig('actua_vs_predicted.png')
plt.show()

# Visualize feature importance
feature_importance = pd.Series(model.feature_importances_,index = housing.feature_names)
feature_importance.sort_values(ascending = False).plot(kind ='bar', figsize = (10,6))
plt.title('Feature Importance in Random Forest Model')
plt.ylabel('Importance')
plt.savefig('Feature_importance.png')
plt.show()


# Example: Predict house value for a new sample
new_house = X_test[0].reshape(1, -1)  # use one test sample as an example
prediction = model.predict(new_house)
print(f"\nPredicted house value for new sample: ${prediction[0]*100000: .2f}")