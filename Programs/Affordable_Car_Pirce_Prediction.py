import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Define the input and output column names
input_cols = ["Year", "Present_Price", "Kms_Driven", "Owner"]
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
output_cols = ["Selling_Price"]

# Function to customize the dataset
def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # Randomly sample a portion of the dataset
    dataframe = dataframe.sample(int(0.95 * len(dataframe)), random_state=int(ord(rand_str[0])))

    # Scale input features
    dataframe.Year = dataframe.Year * ord(rand_str[1]) / 100.0
    dataframe.Present_Price = dataframe.Present_Price * ord(rand_str[2]) / 100.0
    dataframe.Kms_Driven = dataframe.Kms_Driven * ord(rand_str[3]) / 1000.0
    dataframe.Owner = dataframe.Owner * ord(rand_str[4]) / 10.0  # Scale for owners

    # Normalize Selling_Price to be non-negative
    dataframe.Selling_Price = dataframe.Selling_Price.clip(lower=0)  # Ensure no negative selling prices
    
    return dataframe

# Function to convert DataFrame to arrays
def dataframe_to_arrays(dataframe):
    dataframe1 = dataframe.copy(deep=True)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    
    return inputs_array, targets_array

# Load the data
dataframe_raw = pd.read_csv(r"C:\Users\gokul\Downloads\car_data.csv")

# Preprocess data
dataframe = customize_dataset(dataframe_raw, "gokul")
inputs_array, targets_array = dataframe_to_arrays(dataframe)

# Convert inputs and targets to PyTorch tensors
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)

# Create a dataset
dataset = TensorDataset(inputs, targets)

# Split the dataset into training and validation sets
train_ds, val_ds = random_split(dataset, [228, 57])

# Define the batch size
batch_size = 128

# Create data loaders
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# Define the model architecture
class CarsModel(nn.Module):
    def _init_(self):
        super(CarsModel, self)._init_()
        self.linear = nn.Linear(len(input_cols), len(output_cols)) 
        
    def forward(self, xb):
        return self.linear(xb)

    def training_step(self, batch):
        inputs, targets = batch 
        out = self(inputs)          
        loss = F.mse_loss(out, targets)  # Use Mean Squared Error Loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.mse_loss(out, targets)  
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))

# Initialize the model
model = CarsModel()

# Evaluation algorithm
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Fitting algorithm
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs) 
        history.append(result['val_loss'])
    return history

# Define a simple game between buyers and sellers
def simulate_game(num_iterations=100):
    buyer_strategies = np.random.uniform(0.8, 1.2, num_iterations)  # Buyers want to pay between 80% and 120% of the predicted price
    seller_strategies = np.random.uniform(0.9, 1.1, num_iterations)  # Sellers want to sell between 90% and 110% of the predicted price
    predictions = []  # To hold predicted prices

    # Simulate the game
    for _ in range(num_iterations):
        # Randomly select an input sample
        random_index = np.random.randint(0, inputs.shape[0])  # Select a random index
        input_sample = inputs[random_index].unsqueeze(0)  # Get the corresponding input row and add batch dimension
        
        # Get a prediction
        prediction = model(input_sample)  # Get a prediction from the model
        predicted_price = prediction.item()  # Extract the predicted price
        
        # Ensure prediction is non-negative
        predicted_price = max(predicted_price, 0)

        # Calculate buyer's and seller's offered prices
        buyer_offer = predicted_price * buyer_strategies[_]
        seller_offer = predicted_price * seller_strategies[_]

        # Decision making based on offers
        if buyer_offer < seller_offer:
            # Buyer buys at the seller's price
            final_price = seller_offer
        else:
            # No deal
            final_price = 0

        predictions.append(final_price)
    
    return predictions

# Simulate the game
predicted_prices = simulate_game()

# Visualize predicted prices from the game simulation
plt.figure(figsize=(10, 5))
plt.plot(predicted_prices, label='Game Predicted Prices')
plt.xlabel('Iterations')
plt.ylabel('Price')
plt.title('Car Price Predictions from Game Theory Simulation')
plt.legend()
plt.show()

# Function to make a single prediction
def predict_single(input_data, model):
    model.eval()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  
    with torch.no_grad():  
        prediction = model(input_tensor)
    return prediction.numpy()

# Example of making predictions using the model
# Prepare input data for prediction (only include the required input features)
year = 2020
present_price = 500000
kms_driven = 30000
owner = 1

# Prepare input data without the categorical encodings for prediction
input_data = [year, present_price, kms_driven, owner]

# Make a prediction
prediction = predict_single(input_data, model)
print(f"Prediction: {prediction}")