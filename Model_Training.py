from Fashion_MNIST_model import FashionMNIST
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from dataPreprocessing import preprocessedData

x_train,y_train,x_valid,y_valid,_,_ = preprocessedData()

train_dataloader = dataset.load_dataset(x_train,y_train,64)
validation_dataloader = dataset.load_dataset(x_valid,y_valid,64,False) 

epochs = 100
lrate = 0.0015

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inp_dim = x_train.shape[1]
out_dim = 10
hidn_lyrs = 4
neurons_per_lyr = 256
w_decay = 1.5e-4
batch = 64
epochs=100

modelx = FashionMNIST(inp_dim,out_dim,hidn_lyrs,neurons_per_lyr).to(device)
crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelx.parameters(),lr=0.00015,weight_decay=w_decay)


best_accuracy = 0.0 

for epoch in range(epochs):
    modelx.train()
    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        output = modelx(batch_x)
        loss = crit(output, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    modelx.eval()
    tot = 0
    corr = 0
    with torch.no_grad():
        for batch_x, batch_y in validation_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = modelx(batch_x)
            _, prediction = torch.max(output, 1)
            
            tot += batch_y.shape[0]
            corr += (prediction == batch_y).sum().item()
            
    current_accuracy = (corr / tot) * 100
    
    if(epoch%10==0): print(f"Epoch {epoch+1}/{epochs} | Accuracy: {current_accuracy:.2f}%")

    if current_accuracy > best_accuracy:
        print(f"{best_accuracy:.2f}% --> {current_accuracy:.2f}%")
        
        best_accuracy = current_accuracy
        
        torch.save(modelx.state_dict(), 'Fashion_MNIST_TorchModel.pth')