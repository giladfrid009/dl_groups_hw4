import time
from tqdm import tqdm
from typing import Iterable, Any
import logging

import torch
from torch import optim
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning as lit
from lightning.pytorch import LightningDataModule
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch import callbacks, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.utilities import throughput
from lightning.fabric.utilities.seed import seed_everything


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
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class GaussianDataModule(LightningDataModule):
    def __init__(
        self,
        sample_shape: tuple[int, int],
        train_size: int,
        test_size: int,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.sample_shape = sample_shape
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:

        if stage == "fit" or stage is None:
            self.ds_train = GaussianDataset(
                num_samples=self.train_size,
                shape=self.sample_shape,
                var1=1.0,
                var2=0.8,
                static=False,
            )

            self.ds_val = GaussianDataset(
                num_samples=self.test_size,
                shape=self.sample_shape,
                var1=1.0,
                var2=0.8,
                static=True,
            )

        if stage == "test" or stage is None:
            self.ds_test = GaussianDataset(
                num_samples=self.test_size,
                shape=self.sample_shape,
                var1=1.0,
                var2=0.8,
                static=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


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


def get_models(n: int, d: int, model_names: list[str] | None = None) -> Iterable[tuple[nn.Module, str]]:

    # Canonical models
    model_name = "canonical-mlp"
    if model_names is not None and model_name in model_names:
        model = CanonicalModel(create_mlp_model(n, d))
        yield model, model_name

    model_name = "canonical-attn"
    if model_names is not None and model_name in model_names:
        model = CanonicalModel(create_transformer_model(n, d))
        yield model, model_name

    # Symmetry models

    model_name = "symmetry-mlp"
    if model_names is not None and model_name in model_names:
        model = SymmetryModel(
            model=create_mlp_model(n, d),
            perms=list(create_all_permutations(n)),
            chunksize=10,
        )
        yield model, model_name

    model_name = "symmetry-attn"
    if model_names is not None and model_name in model_names:
        model = SymmetryModel(
            model=create_transformer_model(n, d),
            perms=list(create_all_permutations(n)),
            chunksize=10,
        )
        yield model, model_name

    # Sampled symmetry models

    model_name = "symmetry-sampling-mlp"
    if model_names is not None and model_name in model_names:
        num_perms = 10
        model = SymmetryModel(
            model=create_mlp_model(n, d),
            perms=[Permutation(torch.randperm(n)) for _ in range(num_perms)],
            chunksize=10,
        )
        yield model, model_name

    model_name = "symmetry-sampling-attn"
    if model_names is not None and model_name in model_names:
        num_perms = 10
        model = SymmetryModel(
            model=create_transformer_model(n, d),
            perms=[Permutation(torch.randperm(n)) for _ in range(num_perms)],
            chunksize=10,
        )
        yield model, model_name

    # Intrinsic models

    model_name = "intrinsic"
    if model_names is not None and model_name in model_names:
        model = create_invariant_model(n, d)
        yield model, model_name

    # Augmented models

    model_name = "augmented-mlp"
    if model_names is not None and model_name in model_names:
        model = nn.Sequential(
            RandomPermute(),
            create_mlp_model(n, d),
        )
        yield model, model_name

    model_name = "augmented-attn"
    if model_names is not None and model_name in model_names:
        model = nn.Sequential(
            RandomPermute(),
            create_transformer_model(n, d),
        )
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

    for model, model_name in get_models(seq_len, feature_dim, model_names):

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

    for model, model_name in get_models(seq_len, feature_dim, model_names):

        model = model.to(device)

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

    for model, model_name in get_models(seq_len, feature_dim, model_names):

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
    seed: int | None = None,
) -> None:

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    if seed is not None:
        seed_everything(seed, workers=True)

    test_size = 1000

    data = GaussianDataModule(
        sample_shape=(seq_len, feature_dim),
        train_size=train_size,
        test_size=test_size,
        num_workers=4,
    )

    for model, model_name in get_models(seq_len, feature_dim, model_names):

        print()
        print("=" * 100)
        print(f"Running experiment for model {model_name.upper()}")
        print(f"Train size: {train_size}, sequence length: {seq_len}")
        print("=" * 100)

        model = LitModel(model)

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

        trainer = Trainer(
            accelerator="auto",
            strategy="auto",
            devices="auto",
            precision="32-true",
            max_epochs=-1,
            max_time="00:00:30:00",
            callbacks=[early_stop_callback, checkpoint_callback, summary_callback],
            logger=logger,
            log_every_n_steps=50,
            enable_checkpointing=True,
            benchmark=False,
            deterministic=seed is not None,
            fast_dev_run=False,
        )

        # tuner = Tuner(trainer)

        # tuner.scale_batch_size(lightning_model, datamodule=data)

        # tuner.lr_find(lightning_model, datamodule=data)

        trainer.fit(model, datamodule=data)

        trainer.test(model, datamodule=data, ckpt_path="best", verbose=True)
