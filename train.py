import os
import torch
from torch import nn

from dataset.tiny_imagenet import get_dataloaders
from eval import validate
from models.customnet import CustomNet
from utils.prepare_data import prepare_tiny_imagenet

try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%")
    return train_loss, train_accuracy


def main(num_epochs=10, batch_size=32, lr=0.001, momentum=0.9, use_wandb=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    if use_wandb and wandb is not None:
        wandb.init(
            project="aml-lab3-customnet",
            config={"epochs": num_epochs, "batch_size": batch_size, "lr": lr, "momentum": momentum},
        )

    dataset_dir = prepare_tiny_imagenet("data/tiny-imagenet")
    train_loader, val_loader = get_dataloaders(data_root=dataset_dir, batch_size=batch_size)

    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_acc = 0.0
    best_path = os.path.join("checkpoints", "customnet_best.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
            })

    print(f"Best validation accuracy: {best_acc:.2f}%")
    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()