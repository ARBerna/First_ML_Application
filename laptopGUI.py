#make GUI
import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

model = joblib.load("laptop_price_model.pkl")

#GUI window
root = tk.Tk()
root.title("Laptop Price Predictor")
root.geometry("450x500")

title_label = ttk.Label(root, text = "Laptop Price Predictor", font = ("Arial", 18))
title_label.pack(pady = 10)

#input fields
def add_field(label_text):
    label = ttk.Label(root, text = label_text)
    label.pack()
    entry = ttk.Entry(root)
    entry.pack()
    return entry

brand_entry = add_field("Brand:")
processor_entry = add_field("Processor:")
ram_entry = add_field("RAM (GB):")
rom_entry = add_field("Storage (GB):")
gpu_entry = add_field("GPU:")
display_entry = add_field("Display Size (inches):")
os_entry = add_field("Operating System:")

#predict
def predict_price():
    data = pd.DataFrame([{
        "brand": brand_entry.get(),
        "processor": processor_entry.get(),
        "Ram": int(ram_entry.get()),
        "ROM": int(rom_entry.get()),
        "GPU": gpu_entry.get(),
        "display_size": float(display_entry.get()),
        "OS": os_entry.get()
    }])

    prediction = model.predict(data)[0]
    result_label.config(text = f"Predicted Price: {prediction:.2f}")

#button
predict_button = ttk.Button(root, text = "Predict Price", command = predict_price)
predict_button.pack(pady = 20)

#output
result_label = ttk.Label(root, text = "", font = ("Arial", 16))
result_label.pack(pady = 10)

root.mainLoop()