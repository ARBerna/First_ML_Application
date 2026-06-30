import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")

price = df[['price']]

scaler = MinMaxScaler()

priceScaled = scaler.fit_transform(price)

df['PriceNormalized'] = priceScaled

print(df[['price', 'PriceNormalized']].head())