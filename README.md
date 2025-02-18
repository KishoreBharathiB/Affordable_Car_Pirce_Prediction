# Car Price Prediction using Game Theory and Machine Learning

This project leverages machine learning techniques and game theory to predict car prices based on various features like year, present price, kilometers driven, and owner type. The model uses a simple **Linear Regression** network implemented in **PyTorch** to predict the car's selling price based on input features. Additionally, a **game theory simulation** is implemented where the prices predicted by the model are influenced by buyer and seller strategies.

## Features

- **Linear Regression Model**: A basic neural network with a single linear layer to predict car prices.
- **Game Theory Simulation**: Simulates a buyer-seller negotiation where the buyer and seller have different strategies, influencing the final agreed-upon price.
- **Data Preprocessing**: The data is preprocessed by scaling, encoding categorical variables, and handling missing values.
- **Visualization**: Displays the predicted prices from the simulation to analyze price trends over multiple iterations.

## Requirements

Ensure the following Python libraries are installed:

- `torch`
- `pandas`
- `matplotlib`
- `numpy`

## Data
The dataset used in this project is a car sales dataset containing information about cars for sale. The main columns in the dataset are:

 Year: The year of manufacture of the car.
 Present_Price: The current price of the car in the market.
 Kms_Driven: The total kilometers driven by the car.
 Owner: The number of owners the car has had.
 Fuel_Type: Type of fuel the car uses (e.g., Petrol, Diesel).
 Seller_Type: The type of seller (e.g., Individual, Dealer).
 Transmission: The type of transmission (e.g., Manual, Automatic).
 Selling_Price: The price at which the car is being sold.

## Clone the repository and run my project
```
git clone https://github.com/KishoreBharathiB/Affordable_Car_Pirce_Prediction.git
cd programs
python Affordable_Car_Pirce_Prediction.py
```

## Conclusion 
This **README.md** provides a comprehensive guide on how to understand and use the car price prediction project. It covers data preprocessing, model architecture, training, evaluation, the game theory simulation, and instructions on running the code. Let me know if you need any adjustments!

