import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import time


IMAGE_SIZE = (224,224)
# num: how many images in each class do you want to load in sequence
def load_data(num, path):
    images = []
    labels = []
    dataset = path
    start = time.time()
    for folder in os.listdir(dataset):
        label = folder
        subdir = f'{dataset}/{folder}'
        for file in tqdm(os.listdir(subdir)[:num]):
            img_path = os.path.join(subdir, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE) # to match ResNet
            images.append(image)
            labels.append(int(label[-1]))
    images = np.array(images)
    labels = np.array(labels)
    end = time.time()
    print(f'Time taken for data loading: {end - start}')
    return [images, labels]

def extract_samples(dataset, num_samples_per_class, seed=42):
    np.random.seed(seed)
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]

        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Sampling indices for each class
    sampled_indices = []
    for label, indices in class_indices.items():
        if len(indices) >= num_samples_per_class:
            sampled_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))
        else:
            print(f"Not enough samples in class {label}.")

    # Create a subset using sampled indices
    subset = Subset(dataset, sampled_indices)
    return subset

def get_datasets(path, num_each_class, test_ratio, batch_size, image_resize):
    start = time.time()
    data_transforms = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(path, transform=data_transforms)
    # Extract samples
    subset = extract_samples(dataset, num_each_class)

    # Define the size of the test set
    test_size = int(test_ratio * len(subset))
    train_size = len(subset) - test_size

    # Randomly split the dataset into training and test set
    train_dataset, test_dataset = random_split(subset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

    # Create DataLoaders if needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    end = time.time()
    print(f"Time for loading data: {end - start:.2f}s")
    return train_loader, test_loader

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return 100 * correct / total

def model_train(model, optimizer, criterion, device, train_loader, num_epochs):
    start = time.time()
    model = model.to(device)
    # check model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        acc = 0.0

        for images, labels in tqdm(train_loader):
            # Move inputs and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc += calculate_accuracy(outputs, labels)

            except RuntimeError as exception:
              if "out of memory" in str(exception):
                  print('WARNING: out of memory, will pass this')
                  torch.cuda.empty_cache()
                  continue
              else:
                  raise exception
        # Calculate average loss and accuracy over an epoch
        avg_loss = running_loss / len(train_loader)
        avg_acc = acc / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, ACC: {avg_acc:.4f}")

    end = time.time()
    print(f"Training time: {end - start:.2f}s")
    torch.cuda.empty_cache()


def model_test(model, criterion, device, test_loader):
    # test
    model.eval()
    val_running_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            val_acc += calculate_accuracy(outputs, labels)

    avg_val_loss = val_running_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader)

    print(f'Test Loss: {avg_val_loss:.4f}, Test Acc: {avg_val_acc:.2f}')
    torch.cuda.empty_cache()

class TransitionLayer(nn.Module):
    def __init__(self):
        super(TransitionLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(2048, 3, kernel_size=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, model_a, model_b, transition_layer):
        super(CombinedModel, self).__init__()
        self.model_a = model_a
        self.transition_layer = transition_layer
        self.model_b = model_b

    def forward(self, x):
        x = self.model_a(x)
        x = self.transition_layer(x)
        x = self.model_b(x)
        return x