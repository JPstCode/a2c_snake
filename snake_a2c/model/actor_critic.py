from typing import Tuple
from pathlib import Path

from numpy.typing import NDArray
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.models import save_model, load_model


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, state_size: tuple, action_size: int):
        """Initialize."""
        super().__init__()

        self.common: Sequential
        self.initialize_common(state_size=state_size, action_size=action_size)

        self.actor = layers.Dense(action_size)
        self.critic = layers.Dense(1)

    def initialize_common(self, state_size: tuple, action_size: int):
        """"""
        self.common = Sequential()
        # Convolutional layers
        self.common.add(
            Conv2D(
                filters=16,
                kernel_size=8,
                strides=4,
                input_shape=state_size,
                # padding="default",
                padding="same",
                activation="relu",
                name="C1",
            )
        )
        self.common.add(
            Conv2D(
                filters=32,
                kernel_size=4,
                strides=2,
                # padding="default",
                padding="same",
                activation="relu",
                name="C2",
            )
        )
        # Fully-Connected layers
        self.common.add(Flatten())
        self.common.add(Dense(units=256, activation="relu", name="D1"))

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def save_model(self, output_path: Path, episode: int):
        """Save model in h5 format"""
        self.save_weights(output_path / f"{episode}_a2c.h5")

    def load_model(self, output_path: Path):
        """Load latest weights"""
        files = list(output_path.iterdir())
        episodes = []
        if files:
            for file in files:
                if 'a2c' in file.name:
                    episode = int(file.name.split('_')[0])
                    episodes.append(episode)

            if episodes:
                latest_episode = max(episodes)
                print(f"Loaded weight from episode {latest_episode}!")
                self.load_weights(output_path / f"{latest_episode}_a2c.h5")
