#decision tree code:
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Task 6A - Linear Regression Model
# This program trains a linear regression model to predict laptop prices.

# Load the dataset
df = pd.read_csv("data.csv")

# Display basic dataset information
print("Dataset loaded successfully.")
print("Rows and columns:", df.shape)
print(df.head())

# Target variable
y = df["price"]

# Features used for prediction
X = df.drop("price", axis=1)

# Automatically detect categorical and numeric columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing:
# One-hot encode text columns and keep numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Create the Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results")
print("-------------------------")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


#measuring error for linear regression:
