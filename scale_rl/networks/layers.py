from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from scale_rl.networks.utils import he_normal_init, orthogonal_init
from flax.linen import Dropout

class MLPBlock(nn.Module):
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # sqrt(2) is recommended when using with ReLU activation.
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        return x


class ResidualBlock(nn.Module):
    hidden_dim: int
    dtype: Any
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic):
        original_input = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.hidden_dim * 4, kernel_init=he_normal_init(), dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal_init(), dtype=self.dtype)(x)
        return x + original_input
