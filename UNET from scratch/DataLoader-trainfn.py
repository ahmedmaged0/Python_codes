
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch

trainset = DataLoader(dataset=train_data1, batch_size=2, shuffle=True)

UNETmodel = Unet([64, 128, 256], 3,1)
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(UNETmodel.parameters(), lr=0.001)



def train_fn(dataset, optimizer, criterion):
    
    UNETmodel.train()
    loss_total = 0.0
    for data in tqdm(dataset):
        
        images, msks = data
       
        logits = UNETmodel.forward(images)
        loss_ind = criterion(logits, msks)
        optimizer.zero_grad()
        loss_ind.backward()
        optimizer.step()
        loss_total += loss_ind.item()
    
    return loss_total/len(dataset)


EPOCH = 20

for epoch in range(EPOCH):
  train_loss = train_fn(dataset = trainset, optimizer=optimizer, criterion=criterion)

  print(f'Epoch: {epoch+1}  train loss: {train_loss} ')
  
  

