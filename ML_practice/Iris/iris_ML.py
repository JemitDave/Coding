#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,  confusion_matrix, ConfusionMatrixDisplay


# Set random seed for reproducibility
np.random.seed(42)

#Load the Iris dataser
iris = load_iris()
X = iris.data #Features: sepal length, sepal width, petal length, petal width
y = iris.target # Labels: 0(setosa), 1 (versicolor), 2 (virginica)

# Create a pandas DataFrame for the dataset
df = pd.DataFrame(data = X, columns = iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)
print("\nIris Dataset (first 5 rows):")
print(df.head())

# Verify dataset size
print(f"\nTotal samples: {len(X)}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")


# Split data into training (80%) and testing (20%) sets\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Verify split sizes
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

#Initialize and train the Decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Calculte accuract for trainng and test sets
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"\nTraining accuracy: {train_accuracy: .2f}")
print(f"Test accuracy: {test_accuracy: .2f}")



# Visualize the datast: Scatter plot of sepal length vs petal lenght
plt.figure(figsize=(8,6))
sn.scatterplot(data = df, x = 'sepal length (cm)', y = 'petal length (cm)', hue = 'species', style ='species', s=100)

plt.title('Sepal Length vs Petal length by Species')
plt.savefig('iris_scatter.png')
plt.show()

#Visualize the results: Confusion matrix for test set
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)
plt.figure(figsize = (8,6))
disp.plot(cmap ='Blues')
plt.title('Confusion Matrx(Test Set)')
plt.savefig("Confusion_matrix.png")
plt.show()


# Visualise the decision tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names = iris.feature_names, class_names = iris.target_names, filled= True)
plt.title('Decision Tree Strucure')
plt.savefig('decision_tree.png')
plt.show()

#Example: Predict species for a new flower
new_flower = np.array([[6.0, 3.0, 4.8, 1.8]]) #Example measurements
prediction = model.predict(new_flower)
print(f"Predicted species for new flower: {iris.target_names[prediction][0]}")
