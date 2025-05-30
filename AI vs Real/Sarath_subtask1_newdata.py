# Basic modules and device
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device=('cuda' if torch.cuda.is_available() else 'cpu')

# Data processing
    # Transformations that I need to apply to data
transformations=transforms.Compose([
    transforms.Resize((224, 224)),               
    # transforms.RandomHorizontalFlip(p=0.2),     
    # transforms.RandomRotation(degrees=5),      
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),                      
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
test_transformations=transforms.Compose([
    transforms.Resize((224, 224)),               
    # transforms.RandomHorizontalFlip(p=0.2),     
    # transforms.RandomRotation(degrees=5),      
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),                      
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
# Hyper parameters
num_epoch=20
batch_size=50
learning_rate=0.0001

# Defining path to data
training_dataset_dir='C:\\Users\\Sarat\\Downloads\\New_Data\\New_Data'
testing_dataset_dir='C:\\Users\\Sarat\\Downloads\\induction-task-2025\\Test_Images'



class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
       
        self.image_folder = image_folder
        self.transform = transform
      
        self.image_files = [f for f in os.listdir(image_folder)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        
        return image, self.image_files[idx]
train_dataset_val = torchvision.datasets.ImageFolder(root=training_dataset_dir, transform=transformations)
# val_dataset = torchvision.datasets.ImageFolder(root=training_dataset_dir, transform=transformations)

train_size = int(0.8 * len(train_dataset_val))
val_size = len(train_dataset_val) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_val, [train_size, val_size])


test_dataset=TestDataset(testing_dataset_dir,transform=test_transformations)#Use pin_memory only if u are running on gpu


# Loading Data..-Seeing jutsu
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

# Model
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel,self).__init__()
        self.h_layer1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1)
        self.h_layer2=nn.Conv2d(in_channels=16,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.h_layer3=nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,padding=1,stride=1)
        self.h_layer4=nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu=nn.LeakyReLU(0.05)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1=nn.Linear(in_features=1024*14*14,out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=2)
        # self.fc3=nn.Linear(in_features=64,out_features=2)

    def forward(self,x):
        out=self.relu(self.pool(self.h_layer1(x)))
        out=self.relu(self.pool(self.h_layer2(out)))
        out=self.relu(self.pool(self.h_layer3(out)))
        out=self.relu(self.pool(self.h_layer4(out)))

        out=out.view(out.size(0),-1)
        # or out=nn.Flatten(out)
        out=self.relu(self.fc1(out))
        out=self.dropout(out)
        out=(self.fc2(out))
        # out=self.dropout(out)
        # out=self.fc3(out)

        return out
        
model=AIModel().to(device)
# Loss-Chakra overflow
criterion = nn.CrossEntropyLoss()

# Optimizer-Controlling chakra
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Training arc
no_of_images=len(train_loader)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)
for epoch in range(num_epoch):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        prev_loss=loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.10f}")

with torch.no_grad():
    model.eval()
    n_correct=0
    n_samples=0

    for i,(images,labels) in enumerate(val_loader):
        images=images.to(device)
        labels=labels.to(device)

        outputs=model(images)

        index,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()

        
    acc = 100.0 * (n_correct / n_samples)
    print(f'The accuracy of network is : {acc:.6f} %')

import csv
import re


class_mapping = {0: "Real", 1: "AI"}

# List to store the predictions
predictions = {}



# Process the test dataset
with torch.no_grad():
    model.eval()
    for images, file_names in test_loader:
        images = images.to(device)

        for i, image in enumerate(images):
            
            # Forward pass through the model for each image
            output = model(image.unsqueeze(0)) 
            _, predicted = torch.max(output, 1)  
            predicted_label = class_mapping[predicted.item()]  # Get label based on class mapping
            

            # Store prediction (file_name, predicted_label) in the dictionary
            predictions[file_names[i]] = predicted_label


def sort_numerically(file_name):
    # Extract numbers from the file name using regex
    match = re.search(r'\d+', file_name)
    return int(match.group()) # To Handle cases with no numbers 

sorted_predictions = dict(sorted(predictions.items(), key=lambda item: sort_numerically(item[0])))
# Write predictions to the CSV file
with open('C:\\Users\\Sarat\\OneDrive\\Desktop\\Cynaptics\\submission32.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Label'])

    # Write each file name and its corresponding label to the CSV
    for file_name, label in sorted_predictions.items():
        file_name_without_ext = os.path.splitext(file_name)[0]
        writer.writerow([file_name_without_ext, label])

print("Predictions saved to 'submission32.csv'.")



