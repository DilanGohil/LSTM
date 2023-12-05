import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# Read the data from excel1.xlsx
df1 = pd.read_excel("d:/excel for python/excel3.xlsx")

# Add a date column as the index starting from April 7, 2010
df1["Date"] = pd.date_range(start="2010-04-07", periods=len(df1))
df1.set_index("Date", inplace=True)

# Frame the inputs as a supervised learning problem
def lstm_super(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    columns, names = list(), list()

    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        columns.append(df.shift(i))
        names += [("var(t-%d)" % i)]

    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        columns.append(df.shift(-i))
        if i == 0:
            names += [("var(t)")]
        else:
            names += [("var(t+%d)" % i)]

    # Put it all together
    final = pd.concat(columns, axis=1)
    final.columns = names
    return final

# Load the values from the training dataset
values = df1["Safety Factor"].values

# Convert all data to float data type
values = values.astype("float32")

# Normalize features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values.reshape(-1, 1))

# Frame the inputs as a supervised learning problem
lstm_input = lstm_super(scaled, 1, 1)

# splitting data into train and test set according to 80:20 policy
train = scaled[:1652]
test = scaled[1652:]

# split the train and test further into inputs represented by X and outputs represented by Y
train_X, train_y = train[:-1], train[1:]
test_X, test_y = test[:-1], test[1:]

# reshape the input to be 3D [samples, timesteps, features] as LSTM requires inputs in 3D format
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Design the LSTM model using the Adam optimizer and mean absolute error (MAE) as the loss function
model = Sequential()
model.add(LSTM(365, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")

# Train the model using 50 epochs
history = model.fit(
    train_X,
    train_y,
    epochs=50,
    batch_size=365,
    validation_data=(test_X, test_y),
    verbose=4,
    shuffle=False,
)



# Save the trained model for future predictions
model.save("safety_factor_model.h5")

# Make predictions for future unknown values using the trained model
future_predictions = model.predict(future_X)

# Invert the scaling for the forecasted predictions
future_predictions = scaler.inverse_transform(future_predictions)

# Plot the training loss and validation loss over epochs
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Convert the NumPy array to a DataFrame
test_predicted = pd.DataFrame({'Predicted Safety Factor': test_predictions.flatten()})



# Save the DataFrame to an Excel file
test_predicted.to_excel('E:/New folder (2)/soil mode/excel for predictions/Pythonpredicted_data5.xlsx', index=False)
