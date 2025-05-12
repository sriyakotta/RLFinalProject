import functools
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import dataclasses

from flax.training import dynamic_scale
from dataclasses import dataclass

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.sac.sac_network import SACActor, SACTemperature, SACEncoder
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.critics import QuantileRegressionCritic
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.agents.tqc.tqc_update import update_actor, update_critic_tqc, update_target_network, update_temperature


@dataclass(frozen=True)
class TQCConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate: float
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate: float
    critic_weight_decay: float
    num_quantiles: int
    num_quantiles_to_drop: int

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_tau: float
    gamma: float
    n_step: int

    mixed_precision: bool


class TQCCritic(nn.Module):
    block_type: str
    num_blocks: int
    hidden_dim: int
    num_quantiles: int
    dtype: Any

    def setup(self):
        self.encoder = SACEncoder(
            block_type=self.block_type,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        self.predictor = QuantileRegressionCritic(num_quantiles=self.num_quantiles, dtype=self.dtype)

    def __call__(self, observations, actions, deterministic: bool):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.encoder(x, deterministic=deterministic)
        return self.predictor(x)


@functools.partial(jax.jit, static_argnames=("observation_dim", "action_dim", "cfg"))
def _init_tqc_networks(
    observation_dim: int,
    action_dim: int,
    cfg: TQCConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    actor = Trainer.create(
        network_def=SACActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations, "deterministic": True, "temperature": 1.0},
        tx=optax.adamw(
            learning_rate=cfg.actor_learning_rate,
            weight_decay=cfg.actor_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    critic = Trainer.create(
        network_def=TQCCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            num_quantiles=cfg.num_quantiles,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": critic_key, "observations": fake_observations, "actions": fake_actions, "deterministic": True},
        tx=optax.adamw(
            learning_rate=cfg.critic_learning_rate,
            weight_decay=cfg.critic_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    target_critic = Trainer.create(
        network_def=TQCCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            num_quantiles=cfg.num_quantiles,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": critic_key, "observations": fake_observations, "actions": fake_actions, "deterministic": True},
        tx=None,
    )

    temperature = Trainer.create(
        network_def=SACTemperature(cfg.temp_initial_value),
        network_inputs={"rngs": temp_key},
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=cfg.temp_weight_decay,
        ),
    )

    return rng, actor, critic, target_critic, temperature


@functools.partial(
    jax.jit,
    static_argnames=("gamma", "n_step", "target_tau", "temp_target_entropy", "num_quantiles_to_drop")
)
def _update_tqc_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    target_tau: float,
    temp_target_entropy: float,
    num_quantiles_to_drop: int,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=False,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic_tqc(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=False,
        num_quantiles_to_drop=num_quantiles_to_drop,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return rng, new_actor, new_critic, new_target_critic, new_temperature, info


class TQCAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: TQCConfig,
    ):
        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim        super().__init__(observation_space, action_space, cfg)

        super(TQCAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        self._cfg = TQCConfig(**cfg)
        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_tqc_networks(self._observation_dim, self._action_dim, self._cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        temperature = 1.0 if training else 0.0
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = actor_sample(self._rng, self._actor, observations, temperature)
        return np.array(actions)

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_tqc_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
            num_quantiles_to_drop=self._cfg.num_quantiles_to_drop,
        )

        for key, value in update_info.items():
            update_info[key] = float(value)
        
        return update_info


@jax.jit
def actor_sample(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist = actor(observations=observations, deterministic=True, temperature=temperature)
    return rng, dist.sample(seed=key)
