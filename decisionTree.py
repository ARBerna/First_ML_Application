#decision tree code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")

label = LabelEncoder()
for col in df.columns:
    df[col] = label.fit_transform(df[col])

X = df.drop("price", axis = 1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

dt_model = DecisionTreeRegressor(max_depth = 5, random_state = 42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("Decision Tree training complete.")

from sklearn.tree import export_text

tree_rules = export_text(dt_model, feature_names = list(X.columns))
print(tree_rules)


#measure error for decision tree:
from sklearn.metrics import mean_absolute_error, mean_squared_error

dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)

print("Decision Tree MAE:", dt_mae)
print("Decision Tree MSE:", dt_mse)