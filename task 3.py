import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Housing.csv")  # Download and use the file from Kaggle

# Optional: Convert categorical data
df['mainroad'] = df['mainroad'].map({'yes':1, 'no':0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished':2, 'semi-furnished':1, 'unfurnished':0})
df['guestroom'] = df['guestroom'].map({'yes':1, 'no':0})

# 2. Feature and target selection
X = df[['area', 'bedrooms', 'bathrooms', 'mainroad']]  # Multiple features
y = df['price']

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 6. Visualize simple linear regression (area vs price)
plt.scatter(df['area'], df['price'], color='blue', label='Actual')
plt.plot(df['area'], model.predict(df[['area', 'bedrooms', 'bathrooms', 'mainroad']]), color='red', label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price - Linear Regression")
plt.legend()
plt.show()

# 7. Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
