from Fashion_MNIST_model import FashionMNIST
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from dataPreprocessing import preprocessedData


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_,_,_,_,x_test,y_test = preprocessedData()
test_dataloader = dataset.load_dataset(x_test,y_test,64,False)

inp_dim = x_test.shape[1]
out_dim = 10
hidn_lyrs = 4
neurons_per_lyr = 256
w_decay = 1.5e-4
batch = 64
epochs=100

trainedModel = FashionMNIST(inp_dim, out_dim, hidn_lyrs, neurons_per_lyr)

trainedModel.load_state_dict(torch.load('Fashion_MNIST_TorchModel.pth'))
trainedModel.to(device)

trainedModel.eval()
tot = 0
corr = 0
with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        output = trainedModel(batch_x)
        _, prediction = torch.max(output, 1)
        
        tot += batch_y.shape[0]
        corr += (prediction == batch_y).sum().item()
        
accuracy = (corr / tot) * 100

print(f"Model Guessed {corr} out of {tot}\n Accuracy-> {accuracy}")