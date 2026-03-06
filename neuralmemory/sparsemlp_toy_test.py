"""Let's try training MNIST on this"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralmemory.sparsemlp import SparseMLP


class MNISTSparseMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        sparsity_dim: int = 32,
        alpha: float = 5.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.sparse_mlp = SparseMLP(
            input_dim=784,  # 28x28 flattened MNIST images
            hidden_dim=hidden_dim,
            output_dim=10,  # 10 classes (digits 0-9)
            sparsity_dim=sparsity_dim,
            alpha=alpha,
            beta=beta,
        )

    def forward(self, x):
        # Flatten the images from (batch, 1, 28, 28) to (batch, 784)
        x = x.view(x.size(0), -1)
        return self.sparse_mlp(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", disable=True)
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        t0 = time.time()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t1 = time.time()

        optimizer.step()

        t2 = time.time()

        print(t1 - t0, t2 - t1)

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        accuracy = 100.0 * correct / total
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{accuracy:.2f}%"})

    return total_loss / len(train_loader), 100.0 * correct / total


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", disable=True)
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accuracy = 100.0 * correct / total
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{accuracy:.2f}%"})

    return total_loss / len(test_loader), 100.0 * correct / total


def main():
    # Configuration
    config = {
        "batch_size": 128,
        "learning_rate": 5e-4,  # Reduced learning rate
        "num_epochs": 10,
        "hidden_dim": 10000,
        "sparsity_dim": 32,
        # "sparsity_dim": 512,
        "alpha": 2.0,  # Reduced alpha to prevent overflow
        "beta": 0.0,  # Centered at 0
    }

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model initialization
    model = MNISTSparseMLP(
        hidden_dim=config["hidden_dim"],
        sparsity_dim=config["sparsity_dim"],
        alpha=config["alpha"],
        beta=config["beta"],
    ).to(device)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        f"SparseMLP config: hidden_dim={config['hidden_dim']}, sparsity_dim={config['sparsity_dim']}"
    )

    # Check initial weight ranges
    with torch.no_grad():
        in_weight_std = model.sparse_mlp.in_weight.weight.std().item()
        out_weight_std = model.sparse_mlp.out_weight.std().item()
        print(
            f"Initial weight std - in_weight: {in_weight_std:.4f}, out_weight: {out_weight_std:.4f}"
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    best_test_acc = 0.0

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_sparse_mlp_mnist.pth")
            print(f"  New best test accuracy: {best_test_acc:.2f}%")

    print("\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")

    # Load best model and do final evaluation
    model.load_state_dict(torch.load("best_sparse_mlp_mnist.pth"))
    final_test_loss, final_test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Final test accuracy: {final_test_acc:.2f}%")


if __name__ == "__main__":
    main()
