import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from modules.data_preprocessing import (
    ArtifactDataset,
    apply_canny,
    apply_morphology,
    black_and_white,
)
from modules.wandb_integration import SweepOptimizer, log_evaluation
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import vit_b_16

import wandb

# wandb.init(project="VisionTransformer")


class VerkehrsschilderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, subfolder in enumerate(["y", "n"]):
            folder_path = os.path.join(self.root_dir, subfolder)
            for image_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)

        canny_image = apply_canny(image, (224, 224))
        morphology_image = apply_morphology(image, (224, 224))
        bw_image = black_and_white(image, (224, 224))

        combined_image = np.concatenate(
            (canny_image, morphology_image, bw_image), axis=-1
        )
        combined_image = Image.fromarray(np.uint8(combined_image))
        if self.transform:
            combined_image = self.transform(combined_image)

        return combined_image, label


transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

"""
dataset = VerkehrsschilderDataset("data", transform=transform)

total_count = len(dataset)
train_count = int(0.7 * total_count)
valid_count = int(0.15 * total_count)
test_count = total_count - train_count - valid_count

train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_count, valid_count, test_count]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""

class_names = ["keine Wartelinie", "Wartelinie"]


class CustomViT(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomViT, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.num_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Identity()
        self.fc1 = nn.Linear(self.num_features, self.num_features)
        self.bn1 = nn.BatchNorm1d(self.num_features)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(self.num_features, 2)

    def forward(self, x):
        x = self.vit(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomViT().to(device)

criterion = nn.CrossEntropyLoss()


def validate_model(model, criterion, valid_loader):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_loss /= len(valid_loader)
    validation_accuracy = 100 * correct / total

    return validation_loss, validation_accuracy


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    num_epochs=25,
    patience=5,
):
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        val_loss, val_accuracy = validate_model(model, criterion, valid_loader)

        wandb.log(
            {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy / 100.0,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy / 100.0,
            }
        )

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return history


def evaluate_model(model, criterion, test_loader, class_names):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix - Vision Transformer")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    f1 = f1_score(all_labels, all_preds)

    log_evaluation(test_loss, accuracy / 100.0, f1)


if __name__ == "__main__":

    optimizer = SweepOptimizer("VisionTransformer", "ViT-Preprocessing", "val_accuracy")
    best_params = optimizer.get_best_parameters()

    with wandb.init(project="VisionTransformer", config=best_params) as run:
        config = wandb.config

        run.name = "ViT-Preprocessing"

        model = CustomViT(dropout_rate=config.dropout).to(device)
        optimizer = optim.Adam(
            model.fc2.parameters(), lr=config.learning_rate, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        """
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.batch_size, shuffle=False
        )
        """

        # Load Data
        training_dataset = ArtifactDataset(
            "silvan-wiedmer-fhgr/VisionTransformer/swissimage-10cm-preprocessing:v1",
            "training-preprocessing.npy",
            run,
            transform,
        )

        training_loader = DataLoader(
            training_dataset, batch_size=config.batch_size, shuffle=True
        )

        validation_dataset = ArtifactDataset(
            "silvan-wiedmer-fhgr/VisionTransformer/swissimage-10cm-preprocessing:v1",
            "validation-preprocessing.npy",
            run,
            transform,
        )

        validation_loader = DataLoader(
            validation_dataset, batch_size=config.batch_size, shuffle=False
        )

        test_dataset = ArtifactDataset(
            "silvan-wiedmer-fhgr/VisionTransformer/swissimage-10cm-preprocessing:v1",
            "test-preprocessing.npy",
            run,
            transform,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )

        history = train_model(
            model,
            criterion,
            optimizer,
            scheduler,
            training_loader,
            validation_loader,
            num_epochs=100,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history["train_accuracy"], label="Train")
        axes[0].plot(history["val_accuracy"], label="Validation")
        axes[0].set_title("Model Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(loc="upper left")

        axes[1].plot(history["train_loss"], label="Train")
        axes[1].plot(history["val_loss"], label="Validation")
        axes[1].set_title("Model Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc="upper left")

        plt.suptitle("Model Training - Vision Transformer", fontsize=16)
        plt.tight_layout()
        wandb.log({"training_plot": wandb.Image(fig)})
        plt.close(fig)

        evaluate_model(model, criterion, test_loader, class_names)

    wandb.finish()
