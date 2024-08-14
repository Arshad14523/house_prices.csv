import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Prepare the features and target variable
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predicting price for a new house
new_house = pd.DataFrame({'SquareFootage': [2000], 'Bedrooms': [3], 'Bathrooms': [2]})
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price[0])
