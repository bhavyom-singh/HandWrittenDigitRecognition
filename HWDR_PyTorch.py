import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class FullyConnectedNetwork(nn.Module):
    def __init__(self, num_input, hidden1_size, hidden2_size, num_classes):
        super(FullyConnectedNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(num_input, hidden1_size)
        self.hidden_layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.output_layer = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        hidden1_out = AF.leaky_relu(self.hidden_layer1(x))
        hidden2_out = AF.leaky_relu(self.hidden_layer2(hidden1_out))
        output = AF.log_softmax(self.output_layer(hidden2_out), dim=-1)
        return output
    
def show_some_digit_images(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()       
        

# Training function
def train_model(model, training_data, loss_func, optimizer, num_epochs, device, CUDA_enabled):
    if (device.type == 'cuda' and CUDA_enabled):
        print("...Modeling using GPU...")        
        model = model.to(device=device)
    else:
        print("...Modeling using CPU...")
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            labels = labels.type(torch.LongTensor)
            images = torch.flatten(images, start_dim=1)
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            optimizer.zero_grad() # set the cumulated gradient to zero
            output = model(images) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            loss.backward() # propagating loss backward
            optimizer.step() # updating all parameters after every iteration through backpropagation
    
# Testing function
def test_model(model, testing_data, device, CUDA_enabled):
    num_samples = 0
    num_correct = 0
    with torch.no_grad():
        model.eval()
        for batch_cnt, (images, labels) in enumerate(testing_data):
            images = torch.flatten(images, start_dim=1)
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            num_samples = num_samples + labels.shape[0]
            num_correct = num_correct + (prediction==labels).sum().item()
        accuracy = num_correct/num_samples
        print("> Number of samples=", num_samples, "number of correct prediction=",num_correct, "accuracy=", accuracy)
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is NOT supported in your machine. Only CPU will be used for computation.")

print("------------------Loading Dataset---------------------------")

digit_dataset = load_digits()

print("------------------Splitting Dataset---------------------------")
X_train, X_test, Y_train, Y_test =  train_test_split(digit_dataset.images, digit_dataset.target,test_size=0.2)

print("------------------Converting data to tensor and creating mini batch---------------------------")
X_train_t = torch.from_numpy(X_train).to(torch.float32)
Y_train_t = torch.from_numpy(Y_train).to(torch.float32)
X_test_t = torch.from_numpy(X_test).to(torch.float32)
Y_test_t = torch.from_numpy(Y_test).to(torch.float32)

train_dataset = TensorDataset(X_train_t, Y_train_t)
test_dataset = TensorDataset(X_test_t, Y_test_t)

mini_batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=True)

num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches for training: ", num_train_batches)
print("> Number of batches for testing: ", num_test_batches)

show_digit_image = True
if show_digit_image:
    show_some_digit_images(digit_dataset)

num_input = 8*8   
num_classes = 10  
num_hidden1 = 40  
num_hidden2 = 20

model=FullyConnectedNetwork(num_input, num_hidden1,num_hidden2, num_classes)

print("> Model parameters")
print(model.parameters)

print ("> Model's state dictionary")
for param_tensor in model.state_dict():
    print(param_tensor, model.state_dict()[param_tensor].size())

loss_func = nn.CrossEntropyLoss()

num_epochs = 10
alpha = 0.09        # learning rate

optimizer = optim.Adam(model.parameters(), lr=alpha)

print("............Training Fully connected network model................")
CUDA_enabled = True # since we are using GPU
train_model(model, train_dataloader, loss_func, optimizer, num_epochs, device, CUDA_enabled)

print("............Testing Fully connected network model................")
test_model(model, test_dataloader, device, CUDA_enabled)

print("............Implementing Forward Propagation manually................")

def manualForwardPropagation(model, testing_data, device, CUDA_enabled):
    hidden1_weight = model.state_dict()["hidden_layer1.weight"]
    hidden1_bias = model.state_dict()["hidden_layer1.bias"]
    hidden2_weight = model.state_dict()["hidden_layer2.weight"]
    hidden2_bias = model.state_dict()["hidden_layer2.bias"]
    output_weight = model.state_dict()["output_layer.weight"]
    output_bias = model.state_dict()["output_layer.bias"]
    total_images = 0
    correct_pred = 0
    for batch_cnt, (images, labels) in enumerate(testing_data):
            total_images = total_images + len(images)
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device)
                labels = labels.to(device)            
                for i in range(len(images)):
                    h1_out = AF.leaky_relu(torch.add(torch.matmul(hidden1_weight, torch.flatten(images[i])), hidden1_bias))
                    h2_out = AF.leaky_relu(torch.add(torch.matmul(hidden2_weight, h1_out), hidden2_bias))
                    output = AF.log_softmax(torch.add(torch.matmul(output_weight, h2_out), output_bias), dim=-1)
                    correct_pred = correct_pred + (int(labels[i])==torch.where(output == max(output))[0][0]).sum().item()
            else:
                for i in range(len(images)):
                    h1_out = AF.leaky_relu(np.add(np.matmul(hidden1_weight, torch.flatten(images[i])), hidden1_bias))
                    h2_out = AF.leaky_relu(np.add(np.matmul(hidden2_weight, h1_out), hidden2_bias))
                    output = AF.log_softmax(np.add(np.matmul(output_weight, h2_out), output_bias), dim=-1)
                    correct_pred = correct_pred + (int(labels[i])==np.where(output == max(output))[0][0]).sum().item()
    accuracy = correct_pred/total_images
    print("> Number of samples=", total_images, "number of correct prediction=",correct_pred, "accuracy=", accuracy)

manualForwardPropagation(model, test_dataloader, device, CUDA_enabled)

iterable_batches = iter(test_dataloader) # making the dataloader iterable
images, labels = next(iterable_batches)   # getting the next image and its corresponding label
print(f"actual image value : {labels[3]}")
predictedVal = model(torch.flatten(images[3].to(device)))
print(f"predicted image value : {torch.where(predictedVal == max(predictedVal))[0][0]}")
plt.imshow(images[3])

print(f"actual image value : {labels[0]}")
predictedVal = model(torch.flatten(images[0].to(device)))
print(f"predicted image value : {torch.where(predictedVal == max(predictedVal))[0][0]}")
plt.imshow(images[0])