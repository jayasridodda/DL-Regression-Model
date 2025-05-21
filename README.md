# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Dodda Jayasri

### Register Number:212222240028

```

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Ai_Brain = Sequential([
Dense(units = 9, activation = 'relu',input_shape = [8]),
Dense(units = 9, activation = 'relu'),
Dense(units = 9, activation = 'relu'),
Dense(units = 1)
])

Ai_Brain.summary()

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('data1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df.head()

dataset1 = pd.DataFrame(rows[1:],columns=rows[0])
dataset1 = dataset1.astype({'input':'int'})
dataset1 = dataset1.astype({'output':'int'})

X = dataset1[['input']].values
y = dataset1[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state=0)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

Ai_Brain = Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])

Ai_Brain.compile(optimizer = 'rmsprop', loss='mse')

Ai_Brain.fit(X_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(Ai_Brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)

Ai_Brain.evaluate(X_test1,y_test)

X_n1=[[10]]

X_n1_1=Scaler.transform(X_n1)

Ai_Brain.predict(X_n1_1)

```

### Dataset Information
![Screenshot 2024-09-01 181712](https://github.com/user-attachments/assets/5414be99-4671-46a6-bc31-466bc1bcaf84)



### OUTPUT
Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/dbe52a19-b867-4d20-8752-e4d3a0bb5c8d)

Best Fit line plot

![image](https://github.com/user-attachments/assets/dc767f7a-bdc2-4d98-8d08-bd64d70f8cbe)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/f43571f8-a110-417f-9d7d-747527cdc1f1)


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
