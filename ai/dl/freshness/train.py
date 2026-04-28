from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
import torch.optim as optim

from config import DATA_ROOT, DEFAULT_FRUIT_TYPE, DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, MODEL_ROOT
from dataset import build_loader
from model import build_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(predicted.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = build_loader(
        "train",
        fruit_type=args.fruit_type,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    valid_loader = build_loader(
        "valid",
        fruit_type=args.fruit_type,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        "test",
        fruit_type=args.fruit_type,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    labels = train_loader.dataset.classes
    model = build_model(args.model_name, num_classes=len(labels), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_model_state = None
    best_path = model_root / f"{args.fruit_type}_{args.model_name}_best.pt"
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} valid_f1={valid_metrics['f1']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            wait = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "labels": labels,
                    "fruit_type": args.fruit_type,
                    "model_name": args.model_name,
                    "model_version": args.model_version,
                    "image_size": args.image_size,
                    "valid_metrics": valid_metrics,
                },
                best_path,
            )
            print(f"saved best checkpoint: {best_path}")
        else:
            wait += 1

        if wait >= args.patience:
            print("Early stopping")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1']:.4f}"
    )

    return best_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Harvest Slot freshness classifier.")
    parser.add_argument("--fruit-type", default=DEFAULT_FRUIT_TYPE)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights if available.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
