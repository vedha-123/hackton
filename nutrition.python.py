import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data - food and corresponding nutritional values
data = {
    'food': ['apple', 'banana', 'carrot', 'chicken', 'salmon'],
    'calories': [95, 105, 41, 335, 206],
    'protein': [0.5, 1.3, 0.9, 25, 22],
    'carbs': [25, 27, 10, 0, 0],
    'fat': [0.3, 0.3, 0.2, 14, 13]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Convert food names to numeric values (this is a simplification, in practice you'd use more advanced methods)
df['food_id'] = pd.factorize(df['food'])[0]

# Features (food_id) and target values (nutritional values)
X = df[['food_id']]  # Using food_id as a feature
y = df[['calories', 'protein', 'carbs', 'fat']]  # Nutritional values as targets

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (using Linear Regression here for simplicity)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict the nutritional values for a new food item (e.g., 'egg')
new_food = pd.DataFrame({'food_id': pd.factorize(['egg'])[0]})
predicted_nutrition = model.predict(new_food)

print(f"Predicted nutritional values for 'egg': {predicted_nutrition}")
