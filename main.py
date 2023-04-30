# Ask the user to input the path to their dataset
import pandas as pd

path=r"C:\Users\Derrick Baalaboore\Desktop\Iris-Flower-Classification-Dataset-main\IRIS.csv"
df = pd.read_csv(path)

# Separate the features and target variables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model's performance on the testing set
from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Allow the user to input new data for prediction
import numpy as np

sepal_length = float(input("Enter sepal length (in cm): "))
sepal_width = float(input("Enter sepal width (in cm): "))
petal_length = float(input("Enter petal length (in cm): "))
petal_width = float(input("Enter petal width (in cm): "))

new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred = knn.predict(new_data)

print(f"Predicted species: {pred[0]}")
