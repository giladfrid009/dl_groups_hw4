import math
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


def get_training_models(n: int, d: int) -> Iterable[tuple[nn.Module, str]]:

    # Canonical models

    model = CanonicalModel(create_mlp_model(n, d))
    model_name = "canonical-mlp"
    yield model, model_name

    model = CanonicalModel(create_transformer_model(n, d))
    model_name = "canonical-attn"
    yield model, model_name

    # Symmetry models

    """ shift_perm = Permutation((torch.arange(n) + 1) % n)
    model = SymmetryModel(
        model=create_mlp_model(n, d),
        perm_creator=lambda: create_permutations_from_generators([shift_perm]),
        chunksize=10,
    )
    model_name = "symmetry-mlp"
    yield model, model_name

    model = SymmetryModel(
        model=create_transformer_model(n, d),
        perm_creator=lambda: create_permutations_from_generators([shift_perm]),
        chunksize=10,
    )
    model_name = "symmetry-attn"
    yield model, model_name """

    # Sampled symmetry models

    num_perms = 10
    model = SymmetryModel(
        model=create_mlp_model(n, d),
        perm_creator=lambda: (Permutation(torch.randperm(n)) for _ in range(num_perms)),
        chunksize=10,
    )
    model_name = "symmetry-sampling-mlp"
    yield model, model_name

    model = SymmetryModel(
        model=create_transformer_model(n, d),
        perm_creator=lambda: (Permutation(torch.randperm(n)) for _ in range(num_perms)),
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


def run_experiments(train_size: int, seq_len: int, device: torch.device | None = None) -> None:

    if device is None:
        device = torch.device("cpu")

    feature_dim = 5
    test_size = 1000

    ds_train = GaussianDataset(num_samples=train_size, shape=(seq_len, feature_dim), var1=1.0, var2=0.8, static=False)
    ds_test = GaussianDataset(num_samples=test_size, shape=(seq_len, feature_dim), var1=1.0, var2=0.8, static=True)

    dl_train = DataLoader(dataset=ds_train, batch_size=32, shuffle=False)
    dl_test = DataLoader(dataset=ds_test, batch_size=32, shuffle=False)

    for model, model_name in get_training_models(seq_len, feature_dim):

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

        print(f"Training complete. Test loss: {fit_result.test_loss[-1]}")

        invariant = test_invariant(model, torch.randn(5, seq_len, feature_dim), device=device)

        print(f"Model {model_name} is invariant: {invariant}")

        print("#################\n\n")
