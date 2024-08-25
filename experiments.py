import time
from tqdm.auto import tqdm
from typing import Iterable, Any

import torch
from torch import optim
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning as lit
from lightning.pytorch import callbacks, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.utilities import throughput

from src.gaussian_dataset import GaussianDataset
from src.layers import LinearEquivariant, LinearInvariant, PositionalEncoding
from src.permutation import Permutation, RandomPermute, create_all_permutations, create_permutations_from_generators
from src.models import SymmetryModel, CanonicalModel, test_invariant, test_equivariant


MODEL_NAMES = [
    "canonical-mlp",
    "canonical-attn",
    "symmetry-mlp",
    "symmetry-attn",
    "symmetry-sampling-mlp",
    "symmetry-sampling-attn",
    "intrinsic",
    "augmented-mlp",
    "augmented-attn",
]


class LitModel(lit.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        y_hat: Tensor = self.model(x)
        return y_hat.flatten()

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = (y_hat.round() == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = (y_hat.round() == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def create_mlp_model(n: int, d: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=n * d, out_features=10 * d),
        nn.ReLU(),
        nn.Linear(in_features=10 * d, out_features=10 * d),
        nn.ReLU(),
        nn.Linear(in_features=10 * d, out_features=1),
        nn.Sigmoid(),
    )


def create_transformer_model(n: int, d: int) -> nn.Module:
    return nn.Sequential(
        PositionalEncoding(d_model=d, max_len=n),
        nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(batch_first=True, d_model=d, nhead=1),
            norm=nn.LayerNorm(normalized_shape=d),
            num_layers=1,
        ),
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=n * d, out_features=1),
        nn.Sigmoid(),
    )


def create_invariant_model(n: int, d: int) -> nn.Module:
    return nn.Sequential(
        LinearEquivariant(in_channels=d, out_channels=10),
        nn.ReLU(),
        LinearEquivariant(in_channels=10, out_channels=10),
        nn.ReLU(),
        LinearInvariant(in_channels=10, out_channels=1),
        nn.BatchNorm1d(1),
        nn.Sigmoid(),
    )


def get_models(n: int, d: int) -> Iterable[tuple[nn.Module, str]]:

    # Canonical models

    model = CanonicalModel(create_mlp_model(n, d))
    model_name = "canonical-mlp"
    yield model, model_name

    model = CanonicalModel(create_transformer_model(n, d))
    model_name = "canonical-attn"
    yield model, model_name

    # Symmetry models

    # commented out since the model is too slow, so we dont want to run experiments on it
    # additionally, even creating the permutations is very slow, because there are n! permutations

    """ model = SymmetryModel(
        model=create_mlp_model(n, d),
        perms=list(create_all_permutations(n)),
        chunksize=10,
    )
    model_name = "symmetry-mlp"
    yield model, model_name """

    """ model = SymmetryModel(
        model=create_transformer_model(n, d),
        perms=list(create_all_permutations(n)),
        chunksize=10,
    )
    model_name = "symmetry-attn"
    yield model, model_name """

    # Sampled symmetry models

    num_perms = 10
    model = SymmetryModel(
        model=create_mlp_model(n, d),
        perms=[Permutation(torch.randperm(n)) for _ in range(num_perms)],
        chunksize=10,
    )
    model_name = "symmetry-sampling-mlp"
    yield model, model_name

    model = SymmetryModel(
        model=create_transformer_model(n, d),
        perms=[Permutation(torch.randperm(n)) for _ in range(num_perms)],
        chunksize=10,
    )
    model_name = "symmetry-sampling-attn"
    yield model, model_name

    # Intrinsic models

    model = create_invariant_model(n, d)
    model_name = "intrinsic"
    yield model, model_name

    # Augmented models

    model = nn.Sequential(
        RandomPermute(),
        create_mlp_model(n, d),
    )
    model_name = "augmented-mlp"
    yield model, model_name

    model = nn.Sequential(
        RandomPermute(),
        create_transformer_model(n, d),
    )
    model_name = "augmented-attn"
    yield model, model_name


def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_inference_time(model: nn.Module, input: Tensor, device: torch.device, repeats: int) -> float:
    model = model.to(device)
    input = input.to(device)
    model.eval()

    with torch.inference_mode():

        # Warm-up runs
        for _ in range(10):
            _ = model(input)

        min_time = float("inf")

        for _ in tqdm(range(repeats), leave=False):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            output = model(input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            exec_time = end_time - start_time
            min_time = min(min_time, exec_time)

    return min_time


def measure_training_time(model: nn.Module, input: Tensor, device: torch.device, repeats: int) -> float:
    model = model.to(device)
    input = input.to(device)
    model.train()

    with torch.enable_grad():

        # Warm-up runs
        for _ in range(10):
            model.zero_grad()
            output: Tensor = model(input)
            output.sum().backward()

        min_time = float("inf")

        for _ in tqdm(range(repeats), leave=False):
            model.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            output = model(input)
            output.sum().backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            exec_time = end_time - start_time
            min_time = min(min_time, exec_time)

    return min_time


def run_time_benchmarks(
    seq_len: int,
    feature_dim: int,
    device: torch.device | None = None,
    input_batch: int = 32,
    repeats: int = 1000,
    model_names: list[str] = None,
) -> None:
    if device is None:
        device = auto_device()

    input = torch.randn(input_batch, seq_len, feature_dim).to(device)

    for model, model_name in get_models(seq_len, feature_dim):

        if model_names is not None and model_name not in model_names:
            continue

        inference_time = measure_inference_time(model=model, input=input, device=device, repeats=repeats)
        training_time = measure_training_time(model=model, input=input, device=device, repeats=repeats)
        print(f"Model {model_name} inference time: {inference_time:.8f} s")
        print(f"Model {model_name} training time : {training_time:.8f} s")


def run_flops_benchmarks(
    seq_len: int,
    feature_dim: int,
    device: torch.device | None = None,
    input_batch: int = 32,
    model_names: list[str] = None,
) -> None:

    if device is None:
        device = auto_device()

    input = torch.randn(input_batch, seq_len, feature_dim).to(device)

    for model, model_name in get_models(seq_len, feature_dim):

        if model_names is not None and model_name not in model_names:
            continue

        model_fwd = lambda: model(input)
        fwd_flops = throughput.measure_flops(model, model_fwd)

        model_loss = lambda y: y.sum()
        fwd_and_bwd_flops = throughput.measure_flops(model, model_fwd, model_loss)

        print(f"Model {model_name} forward FLOPs: {fwd_flops:.2e}")
        print(f"Model {model_name} forward and backward FLOPs: {fwd_and_bwd_flops:.2e}")


def run_invariance_tests(
    seq_len: int,
    feature_dim: int,
    device: torch.device | None = None,
    model_names: list[str] = None,
) -> None:

    if device is None:
        device = auto_device()

    input = torch.randn(5, seq_len, feature_dim).to(device)

    for model, model_name in get_models(seq_len, feature_dim):

        if model_names is not None and model_name not in model_names:
            continue

        invariant = test_invariant(
            model=model,
            input=input,
            device=device,
            test_rounds=10,
            tolerance=1e-5,
        )

        print(f"Model {model_name} is invariant: {invariant}")

    equiv_layer = LinearEquivariant(in_channels=feature_dim, out_channels=feature_dim)

    equivariant = test_equivariant(
        model=equiv_layer,
        input=input,
        device=device,
        test_rounds=10,
        tolerance=1e-5,
    )

    print(f"Layer LinearEquivariant is equivariant: {equivariant}")

    inv_layer = LinearInvariant(in_channels=feature_dim, out_channels=1)

    invariant = test_invariant(
        model=inv_layer,
        input=input,
        device=device,
        test_rounds=10,
        tolerance=1e-5,
    )

    print(f"Layer LinearInvariant is invariant: {invariant}")


def run_experiments(
    seq_len: int,
    feature_dim: int,
    train_size: int,
    model_names: list[str] = None,
) -> None:

    test_size = 1000

    ds_train = GaussianDataset(num_samples=train_size, shape=(seq_len, feature_dim), var1=1.0, var2=0.8, static=False)
    ds_val = GaussianDataset(num_samples=test_size, shape=(seq_len, feature_dim), var1=1.0, var2=0.8, static=True)
    ds_test = GaussianDataset(num_samples=test_size, shape=(seq_len, feature_dim), var1=1.0, var2=0.8, static=True)

    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=32,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    for model, model_name in get_models(seq_len, feature_dim):
        if model_names is not None and model_name not in model_names:
            continue

        lightning_model = LitModel(model)

        logger = TensorBoardLogger(save_dir=f"lightning_logs", name=f"train{train_size}_seq{seq_len}/{model_name}")

        early_stop_callback = callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=200,
            strict=True,
            check_finite=True,
        )

        checkpoint_callback = callbacks.ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

        summary_callback = callbacks.RichModelSummary(max_depth=3)

        progress_callback = callbacks.RichProgressBar()

        trainer = Trainer(
            accelerator="auto",
            strategy="auto",
            devices="auto",
            precision="32-true",
            max_epochs=1,
            max_time="00:00:30:00",
            callbacks=[early_stop_callback, checkpoint_callback, summary_callback, progress_callback],
            logger=logger,
            log_every_n_steps=50,
            enable_checkpointing=True,
            benchmark=True,
            profiler="simple",
            deterministic=False,
            fast_dev_run=False,
        )

        trainer.fit(lightning_model, dl_train, dl_val)

        trainer.test(lightning_model, dl_test, ckpt_path="best", verbose=True)
