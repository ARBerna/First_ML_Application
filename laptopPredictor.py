import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("laptop_price_model.pkl")

# Load dataset
df = pd.read_csv("data.csv")

# Recreate SAME encoding used during training
label = LabelEncoder()
for col in df.columns:
    df[col] = label.fit_transform(df[col])

# Use first row as template (ensures correct column order)
input_df = df.iloc[[0]].copy()

# Helper function to ask user for input with options
def ask_user(column_name, options):
    print(f"\nChoose a {column_name} from the list below:")
    for opt in options:
        print("-", opt)
    choice = input(f"Enter your choice for {column_name}: ")
    return choice

# Get options from original CSV (not encoded)
raw_df = pd.read_csv("data.csv")

# Ask user for each field
brand = ask_user("brand", raw_df["brand"].unique())
processor = ask_user("processor", raw_df["processor"].unique())
Ram = ask_user("Ram", raw_df["Ram"].unique())
ROM = ask_user("ROM", raw_df["ROM"].unique())
GPU = ask_user("GPU", raw_df["GPU"].unique())
display_size = ask_user("display_size", raw_df["display_size"].unique())
OS = ask_user("OS", raw_df["OS"].unique())

# Apply SAME encoding logic
input_df["brand"] = label.fit_transform([brand])[0]
input_df["processor"] = label.fit_transform([processor])[0]
input_df["Ram"] = label.fit_transform([Ram])[0]
input_df["ROM"] = label.fit_transform([ROM])[0]
input_df["GPU"] = label.fit_transform([GPU])[0]
input_df["display_size"] = label.fit_transform([display_size])[0]
input_df["OS"] = label.fit_transform([OS])[0]

# Drop price column if present
if "price" in input_df.columns:
    input_df = input_df.drop("price", axis=1)

# Predict
prediction = model.predict(input_df)[0]

print("\nPredicted Laptop Price:", prediction)
