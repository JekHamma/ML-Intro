import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net() # Create the neural network
criterion = nn.MSELoss() # Define the loss function
optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=.99) # Define the optimizer


input_data, target_data = [], []
for x in range(-50, 50):
    i = x/5
    input_data.append([i])
    target_data.append([np.exp(i)])

input_data = np.array(input_data) # Define the input data
target_data = np.array(target_data) # Define the target data

# Train the neural network
for epoch in range(50000):
    optimizer.zero_grad() # zero the gradient buffers
    input_tensor = torch.Tensor(input_data) # convert input data to a tensor
    target_tensor = torch.Tensor(target_data) # convert target data to a tensor
    output_tensor = net(input_tensor) # get the output from the neural network
    loss = criterion(output_tensor, target_tensor) # calculate the loss
    loss.backward() # backpropagate the loss
    optimizer.step() # update the weights
    if epoch % 1000 == 0:
        print('Epoch: ', epoch, '\tLoss: ', loss.item()) # print the loss every 1000 epochs

print('Finished Training')
        
# Test the neural network
input_value = []
for x in range(-500, 500):
    input_value.append([x/100])
    
deployment_input_data = np.array(input_value) # Define the input data for deployment
input_tensor = torch.Tensor(deployment_input_data) # convert input data to a tensor
output_tensor = net(input_tensor) # get the output from the neural network
output_values = output_tensor.detach().numpy() # convert the output tensor to a numpy array
new = np.exp(deployment_input_data)

fig, ax = plt.subplots()
ax.plot(deployment_input_data, new, 'o', label='Expected')
ax.plot(deployment_input_data, abs(np.exp(deployment_input_data) - output_values), label='Error')
ax.plot(deployment_input_data, output_values, 'o', label='NN')
plt.legend()
plt.grid()
plt.show()