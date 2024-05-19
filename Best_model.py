import torch
import os
from torchvision import models, transforms, datasets
from util import model_test
from torch.utils.data import DataLoader
device = torch.device("cuda")
from hyperparameters import(
        save_path,
        learning_rate,
        batch_size,
        num_epochs,
        image_resize,
        num_each_class,
        val_ratio,
        test_ratio)
PATH = 'sfddd/test'
# Create DataLoaders if needed
data_transforms = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(PATH, transform=data_transforms)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ResNet50
model = models.resnet50(weights='DEFAULT')
name = 'models/resnet50.pth'

# Modify final fully connected layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)

# load model
model.load_state_dict(torch.load(name))

# Define loss function
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

pred, true = model_test(model, criterion, device, test_loader)
print(pred)
print(true)