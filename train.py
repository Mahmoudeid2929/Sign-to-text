# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your data
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Separate features and labels
train_x = train.drop('label', axis=1)
train_y = train['label']
test_x = test.drop('label', axis=1)
test_y = test['label']

# Create a model (this is a simple logistic regression model)
model = LogisticRegression()

# Train the model
model.fit(train_x, train_y)

# Test the model
predictions = model.predict(test_x)

# Calculate the accuracy
accuracy = accuracy_score(test_y, predictions)

# Print the predictions and the accuracy
print("Predictions:", predictions)
print("Accuracy:", accuracy*100, "%")
