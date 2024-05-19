import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50
from sklearn.metrics import precision_score, recall_score
from transformers import ViTModel, ViTConfig
import os
from hyperparameters import save_path
# import bestPerformance

def load_data(num, val_ratio, test_ratio, path, batch_size, image_resize, seed=42):
    np.random.seed(seed)

    start = time.time()
    data_transforms = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(path, transform=data_transforms)
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]

        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Sampling indices for each class
    sampled_indices = []
    for label, indices in class_indices.items():
        if len(indices) >= num:
            sampled_indices.extend(np.random.choice(indices, num, replace=False))

        else:
            print(f"Not enough samples in class {label}.")

    np.random.shuffle(sampled_indices)
    dataset_size = len(sampled_indices)

    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    print(f'Train size: {train_size}, Val size: {val_size}, Test size: {test_size}')

    # Create a subset using sampled indices
    train_indices = sampled_indices[:train_size]
    val_indices = sampled_indices[train_size:train_size + val_size]
    test_indices = sampled_indices[train_size + val_size:]

    train_subset, val_subset, test_subset = Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
    train_loader, val_loader, test_loader = (DataLoader(train_subset, batch_size=batch_size, shuffle=True),
                                             DataLoader(val_subset, batch_size=batch_size, shuffle=True),
                                             DataLoader(test_subset, batch_size=batch_size, shuffle=True))
    end = time.time()
    print(f"Time for loading data: {end - start:.2f}s")
    return train_loader, val_loader, test_loader

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return 100 * correct / total

def model_train(model, optimizer, criterion, device, train_loader, val_loader, num_epochs, name, load=False):
    path = os.path.join(save_path, name + '.pth')
    if load:
        try:
            model.load_state_dict(torch.load(path))
            # return bestPerformance.name
        except:
            print('No such model')
    else:
        start = time.time()

        # best_acc = bestPerformance.name if bestPerformance.name else 0.0
        train_acc, train_loss = [], []
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

                    cur_acc = calculate_accuracy(outputs, labels)
                    train_acc.append(cur_acc)
                    train_loss.append(loss.item())
                    running_loss += loss.item()
                    acc += cur_acc

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

            # validate
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_acc += calculate_accuracy(outputs, labels)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)

            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}')

        # if avg_acc > best_acc:
        #     torch.save(model.state_dict(), path)
        #     print(f'New best model saved with accuracy: {avg_acc}%')
        #
        #     # Write the best accuracy to a file
        #     with open('bestPerformance.py', 'r') as file:
        #         lines = file.readlines()
        #     for line in lines:
        #         # Replace the name identifier if it's in the line
        #         if f"{name}_" in line:
        #             updated_line = line.replace(f"{name}_", f"{new_name}_")
        #             updated_lines.append(updated_line)
        #         else:
        #             updated_lines.append(line)
        #
        #
        #
        #         file.write(f'{name} = {avg_acc}\n')
        #         file.write(f'{name}_acc = {train_acc}\n')
        #         file.write(f'{name}_loss = {train_loss}\n')

        end = time.time()
        print(f"Training time: {end - start:.2f}s")
        torch.cuda.empty_cache()
        return train_acc, train_loss


def metrics_plot(pred, true, acc, loss, interval, model_name):
    plt.figure(figsize=(21, 6))

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(true, pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    epoch_indices = range(0, len(acc), interval)
    # Accuracy Plot
    plt.subplot(1, 3, 2)

    sampled_acc = [acc[i] for i in epoch_indices]
    plt.plot(sampled_acc)
    plt.title('Training Accuracy over Batches')
    plt.xlabel(f'Per {interval} Batches')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 3, 3)

    sampled_loss = [loss[i] for i in epoch_indices]
    plt.plot(sampled_loss)
    plt.title('Training Loss over Batches')
    plt.xlabel(f'Per {interval} Batches')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.suptitle(model_name, fontsize=16)  # Adjust the font size and position as needed
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def model_test(model, criterion, device, test_loader):
    # test
    model = model.to(device)
    model.eval()
    val_running_loss = 0.0
    val_acc = 0.0
    pred, true = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pred.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            val_acc += calculate_accuracy(outputs, labels)

    avg_val_loss = val_running_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader)
    pred_np = np.array(pred)
    true_np = np.array(true)

    precision = precision_score(true_np, pred_np, average='macro')
    recall = recall_score(true_np, pred_np, average='macro')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f'Test Loss: {avg_val_loss:.4f}, Test Acc: {avg_val_acc:.2f}')
    torch.cuda.empty_cache()
    return pred, true

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


class ResNetViT(nn.Module):
    def __init__(self, num_classes):
        super(ResNetViT, self).__init__()
        # Load a pre-trained ResNet and remove the last FC layer
        self.resnet = resnet50(weights='DEFAULT')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # ViT configuration - adjust parameters according to your needs
        config = ViTConfig(image_size=7,  # Since we're using the output from ResNet
                           patch_size=1,  # Since patches are the 1x1 output feature maps
                           num_channels=2048,  # Number of input channels from ResNet
                           num_hidden_layers=6,  # Number of transformer layers
                           num_attention_heads=8,  # Number of attention heads
                           hidden_size=512,  # Dimensionality of transformer layers
                           num_labels=num_classes)
        self.vit = ViTModel(config=config)

        # A classifier head
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)  # Shape: [batch_size, 2048, 7, 7]
        outputs = self.vit(x)  # Correctly pass embeddings
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits
