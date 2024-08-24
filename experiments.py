import math
import time
from tqdm.auto import tqdm
from typing import Iterable
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from src.gaussian_dataset import GaussianDataset
from src.training import BinaryTrainer
from src.layers import LinearEquivariant, LinearInvariant, PositionalEncoding
from src.train_results import FitResult
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
        device = torch.device("cpu")

    input = torch.randn(input_batch, seq_len, feature_dim).to(device)

    for model, model_name in get_models(seq_len, feature_dim):

        if model_names is not None and model_name not in model_names:
            continue

        inference_time = measure_inference_time(model=model, input=input, device=device, repeats=repeats)
        training_time = measure_training_time(model=model, input=input, device=device, repeats=repeats)
        print(f"Model {model_name} inference time: {inference_time:.8f} seconds")
        print(f"Model {model_name} training time : {training_time:.8f} seconds")


def run_invariance_tests(
    seq_len: int,
    feature_dim: int,
    device: torch.device | None = None,
    model_names: list[str] = None,
) -> None:
    if device is None:
        device = torch.device("cpu")

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
    device: torch.device | None = None,
    model_names: list[str] = None,
) -> None:

    if device is None:
        device = torch.device("cpu")

    test_size = 1000

    ds_train = GaussianDataset(
        num_samples=train_size,
        shape=(seq_len, feature_dim),
        device=device,
        var1=1.0,
        var2=0.8,
        static=False,
    )

    ds_test = GaussianDataset(
        num_samples=test_size,
        shape=(seq_len, feature_dim),
        device=device,
        var1=1.0,
        var2=0.8,
        static=True,
    )

    dl_train = DataLoader(dataset=ds_train, batch_size=32, shuffle=False)
    dl_test = DataLoader(dataset=ds_test, batch_size=32, shuffle=False)

    for model, model_name in get_models(seq_len, feature_dim):

        if model_names is not None and model_name not in model_names:
            continue

        print(f"Training model: {model_name}\n\n")

        trainer = BinaryTrainer(
            model=model,
            criterion=nn.BCELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=device,
            log=True,
            log_dir=f"train{train_size}_seq{seq_len}/{model_name}",
        )

        fit_result = trainer.fit(
            dl_train=dl_train,
            dl_test=dl_test,
            num_epochs=10000,
            print_every=25,
            time_limit=60 * 30,
            early_stopping=200,
        )

        print("\n\n#################")
        print("Training complete.")
        print(f"Final Test loss    : {fit_result.test_loss[-1]}")
        print(f"Final Test accuracy: {fit_result.test_acc[-1]}")
        print("#################\n\n")
