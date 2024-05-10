import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cnn_tpms import datasets, models, train_utils


def plot_and_save_learning_curves(
    logs: Dict[str, List[float]], image_path: str, epochs: int
) -> None:
    plt.plot(logs["mse_loss"], label="Train MSE")
    plt.plot(logs["ce_loss"], label="Train CrossEntropy ")
    plt.plot(logs["mse_test_loss"], label="Test MSE")
    plt.plot(logs["mae_test_loss"], label="Test MAE")
    plt.plot(logs["ce_test_loss"], label="Test CrossEntropy")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(epochs), np.arange(1, epochs + 1))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(image_path)


def fit_epochs(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: Union[str, int],
    test_data_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    mse_alpha: float = 1,
    model_checkpoint: str = "model_epochs.pth",
    early_stopping_limit: int = 5,
) -> Dict[str, Any]:
    gradient_scaler = torch.cuda.amp.GradScaler()
    model.train()
    mse_criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    l1_criterion = torch.nn.L1Loss()
    logs: Dict[str, List[float]] = defaultdict(list)
    num_batches_train = len(train_data_loader)
    best_loss: float = np.inf
    early_stopping_counter = 0
    for i in range(epochs):
        if early_stopping_counter >= early_stopping_limit:
            print(f"Training early stopped after {i+1} epochs")
            break
        running_loss = 0.0
        running_loss_mse = 0.0
        running_loss_l1 = 0.0
        running_loss_ce = 0.0
        print(f"Epoch {i+1} / {epochs}")
        loop = tqdm(train_data_loader, leave=True)
        for data, target in loop:
            data = data.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast():
                predicted = model.forward(data)
                mse_loss = (
                    mse_criterion(predicted.density, target.density) * mse_alpha
                )
                ce_loss = ce_criterion(predicted.label, target.label)
                l1_loss = l1_criterion(predicted.density, target.density)
                loss = mse_loss + ce_loss
                running_loss_mse += mse_loss.item()
                running_loss_l1 += l1_loss.item()
                running_loss_ce += ce_loss.item()
                running_loss += (mse_loss + ce_loss).item()
            optimizer.zero_grad()
            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            loop.set_postfix(
                loss=loss.item(),
                MSE=mse_loss.item(),
                MAE=l1_loss.item(),
                CrossEntropy=ce_loss.item(),
            )

        mse_train_loss = running_loss_mse / num_batches_train
        ce_train_loss = running_loss_ce / num_batches_train
        l1_train_loss = running_loss_l1 / num_batches_train
        train_loss = running_loss / num_batches_train
        logs["loss"].append(train_loss)
        logs["mse_loss"].append(mse_train_loss)
        logs["l1_loss"].append(l1_train_loss)
        logs["ce_loss"].append(ce_train_loss)
        print(f"Train Loss = {train_loss}")
        print(f"MSE Loss = {mse_train_loss}")
        print(f"MAE Loss = {l1_train_loss}")
        print(f"CrossEntropy Loss = {ce_train_loss}")
        print("Running on test data")
        model.eval()
        num_batches = len(test_data_loader)
        running_loss = 0.0
        running_loss_mse = 0.0
        running_loss_l1 = 0.0
        running_loss_ce = 0.0

        with torch.no_grad():
            for data, target in test_data_loader:
                data = data.to(device)
                target = target.to(device)
                predicted = model(data)
                mse_loss = mse_criterion(predicted.density, target.density)
                ce_loss = ce_criterion(predicted.label, target.label)
                l1_loss = l1_criterion(predicted.density, target.density)
                running_loss_l1 += l1_loss.item()
                running_loss_mse += mse_loss.item()
                running_loss_ce += ce_loss.item()
                running_loss += (mse_loss + ce_loss).item()

        mse_test_loss = running_loss_mse / num_batches
        ce_test_loss = running_loss_ce / num_batches
        test_loss = running_loss / num_batches
        l1_test_loss = running_loss_l1 / num_batches
        if test_loss < best_loss:
            best_loss = test_loss
            train_utils.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=i + 1,
                save_path=model_checkpoint,
            )
            print("Checkpoint saved")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        logs["test_loss"].append(test_loss)
        logs["mse_test_loss"].append(mse_test_loss)
        logs["ce_test_loss"].append(ce_test_loss)
        logs["mae_test_loss"].append(l1_test_loss)
        print(f"Test Loss = {test_loss}")
        print(f"MSE Loss = {mse_test_loss}")
        print(f"MAE Loss = {l1_test_loss}")
        print(f"CrossEntropy Loss = {ce_test_loss}")
        writer.add_scalar("Loss/Loss Train", train_loss, i + 1)
        writer.add_scalar("Loss/MSE Train", mse_train_loss, i + 1)
        writer.add_scalar("Loss/MAE Train", l1_train_loss, i + 1)
        writer.add_scalar("Loss/CE Train", ce_train_loss, i + 1)
        writer.add_scalar("Loss/Loss Test", test_loss, i + 1)
        writer.add_scalar("Loss/MSE Test", mse_test_loss, i + 1)
        writer.add_scalar("Loss/MAE Test", l1_test_loss, i + 1)
        writer.add_scalar("Loss/CE Test", ce_test_loss, i + 1)

    return logs


def compute_confusion_matrix(
    model: torch.nn.Module, test_dataloader: DataLoader, device: str | int
) -> Tuple[np.ndarray, float]:
    model = model.to(device)
    model.eval()
    total_predictions = []
    total_labels = []
    accuracy = 0.0
    for data, label in test_dataloader:
        label = label.label
        with torch.no_grad():
            data = data.to(device)
            predictions = model(data).label.argmax(axis=1).to("cpu")
        total_predictions.append(predictions)
        total_labels.append(label)
        accuracy += (predictions == label).type(torch.float).sum().item()
    accuracy /= len(test_dataloader.dataset)
    total_predictions = np.hstack(total_predictions)
    total_labels = np.hstack(total_labels)
    conf_matrix = confusion_matrix(total_labels, total_predictions)
    print(f"Total accuracy {accuracy}")
    return conf_matrix, accuracy


def save_confusion_matrix(
    conf_matrix: np.ndarray, labels: Iterable[str], image_path: str
) -> None:
    df_cm = pd.DataFrame(conf_matrix, index=list(labels), columns=list(labels))
    plt.figure(figsize=(10, 10))
    cm = sn.heatmap(df_cm, annot=True, fmt="g")
    figure = cm.get_figure()
    figure.savefig(image_path, dpi=400)


def main(
    batch_size: int = 12,
    epochs: int = 20,
    device: int = 0,
    mse_alpha: float = 1,
    dataset_folder: str = "dataset_mesh/",
    training_folder: str | Path = "train_cnn/",
    model_checkpoint: str = "model_epochs.pth",
    num_workers: int = 12,
    voxel_resolution: int = 80,
    early_stopping_limit: int = 5,
) -> None:
    seed = 69
    torch.manual_seed(seed)
    training_folder = Path(training_folder)
    os.makedirs(training_folder, exist_ok=True)
    # Load Dataset
    train_csv_filename = f"{dataset_folder}train_dataset.csv"
    test_csv_filename = f"{dataset_folder}test_dataset.csv"
    train_dataframe = pd.read_csv(train_csv_filename)
    test_dataframe = pd.read_csv(test_csv_filename)
    # Define dataset params
    volume_grid_shape: Tuple[int, int, int] = tuple(
        np.ceil(np.sqrt((3, 3, 3)) * voxel_resolution).astype(int)
    )
    train_dataset = datasets.LatticeStlVolumes(
        train_dataframe,
        voxel_resolution=voxel_resolution,
        volume_grid_shape=volume_grid_shape,
        transform=datasets.rotation_transform,
    )
    test_dataset = datasets.LatticeStlVolumes(
        test_dataframe,
        voxel_resolution=voxel_resolution,
        volume_grid_shape=volume_grid_shape,
        transform=datasets.rotation_transform,
    )
    print(f"Split seed {seed}")
    print(f"Batch size {batch_size}")
    print(f"Epochs {epochs}")
    print(f"Num workers {num_workers}")
    print(f"MSE alpha {mse_alpha}")
    print(f"Dataset folder {dataset_folder}")
    print(f"Voxel resolution {voxel_resolution}")
    print(f"Volume grid shape {volume_grid_shape}")
    print(f"Train dataset size = {len(train_dataset)}")
    print(f"Test dataset size = {len(test_dataset)}")
    # Data loading params

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
    )

    # Model definition
    n_classes = 9
    convolution_model = models.Convolution3dModel(
        input_size=volume_grid_shape, in_channels=1, n_classes=n_classes
    )
    model = models.DensityClassifConvolutionModel(convolution_model)
    # Train parameters
    optimizer = torch.optim.Adam(model.parameters())
    model = model.to(device)
    # Launch training
    with SummaryWriter(log_dir=training_folder / "tensorboard") as writer:
        logs = fit_epochs(
            model,
            writer=writer,
            train_data_loader=train_dataloader,
            test_data_loader=test_dataloader,
            epochs=epochs,
            device=device,
            optimizer=optimizer,
            mse_alpha=mse_alpha,
            model_checkpoint=model_checkpoint,
            early_stopping_limit=early_stopping_limit,
        )

    train_utils.load_checkpoint(model=model, save_path=model_checkpoint)
    print(f"Best model checkpoint loaded {model_checkpoint}")
    # Plot learning curves
    curves_image_filename = training_folder / "curves_loss.png"
    plot_and_save_learning_curves(logs, curves_image_filename, epochs)

    # Save confusion matrix
    labels_name = np.asarray(sorted(train_dataframe["class"].unique()))
    conf_matrix, accuracy = compute_confusion_matrix(
        model, test_dataloader, device
    )
    # Save logs as json
    logs["accuracy"] = accuracy
    logs["batch_size"] = batch_size
    logs["seed"] = seed
    logs["num_workers"] = num_workers
    logs["n_epochs"] = epochs
    logs["mse_alpha"] = mse_alpha
    logs["voxel_resolution"] = voxel_resolution
    logs["train_size"] = len(train_dataset)
    logs["test_size"] = len(test_dataset)
    logs["volume_grid_shape"] = str(volume_grid_shape)
    logs["dataset_folder"] = dataset_folder
    save_confusion_matrix(
        conf_matrix, labels_name, training_folder / "conf_matrix.png"
    )
    with open(training_folder / "logs.json", "w") as fp:
        json.dump(logs, fp)


if __name__ == "__main__":
    fire.Fire(main)
