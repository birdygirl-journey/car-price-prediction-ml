# Car Price Prediction Project
# Day 21–22

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load data
df = pd.read_csv('car_data.csv')
print("Dataset preview:\n", df.head(), "\n")

# Step 2: Define features and target
X = df[['Year', 'Present_Price', 'Kms_Driven', 'Owner']]
y = df['Selling_Price']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Test on new data
new_car = pd.DataFrame({'Year': [2019], 'Present_Price': [12.5], 'Kms_Driven': [10000], 'Owner': [0]})
predicted_price = model.predict(new_car)
print("\nPredicted Selling Price for New Car: $", round(predicted_price[0], 2))
