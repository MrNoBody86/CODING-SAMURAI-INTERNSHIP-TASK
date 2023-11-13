import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets

# Step 1: Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame to view the data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column (species)
df['species'] = iris.target_names[iris.target]

# Step 2: Data Exploration
# Visualize data relationships
sns.pairplot(df, hue='species')
plt.show()

# Step 3: Data Preprocessing
# Check for missing values (not needed for Iris dataset as it's usually clean)
# Encode the categorical target variable (species)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split the dataset into a training set and a testing set
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Model Selection
# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Adjust max_iter as needed

# Step 5: Model Training
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report (includes precision, recall, F1-score)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Prediction
# Example usage: Predict the species for a new data point
new_data_point = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own measurements
predicted_species = model.predict(new_data_point)
print(f"Predicted Flower Species: {iris.target_names[predicted_species][0]}")

# Step 8: Visualization (Confusion Matrix)
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
