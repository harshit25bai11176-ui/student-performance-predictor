import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

# Features and target
X = data[["hours", "attendance"]]
y = data["marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))

# Prediction
prediction = model.predict([[hours, attendance]])

# Pass/Fail logic
result = "Pass" if prediction[0] >= 40 else "Fail"

print(f"Predicted Marks: {prediction[0]:.2f}")
print(f"Result: {result}")



import matplotlib.pyplot as plt

plt.scatter(data["hours"], data["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()