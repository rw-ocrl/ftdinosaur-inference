"""Helper functions.

Partially adapted from the oclf library under Apache License 2.0.

See: https://github.com/amazon-science/object-centric-learning-framework/
Original file: https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/neural_networks/convenience.py
"""

from typing import Callable, List, Optional, Union

from torch import nn


def get_activation_fn(
    name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None
):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    """Build MLP."""
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def build_two_layer_mlp(
    input_dim,
    output_dim,
    hidden_dim,
    activation_fn: str = "relu",
    initial_layer_norm: bool = False,
    residual: bool = False,
):
    """Build a two layer MLP, with optional initial layer norm."""
    return build_mlp(
        input_dim,
        output_dim,
        [hidden_dim],
        activation_fn=activation_fn,
        initial_layer_norm=initial_layer_norm,
        residual=residual,
    )


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)
