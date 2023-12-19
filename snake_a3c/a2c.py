""""""

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


class ActorCriticModel(tf.keras.Model):
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
        # self.save_weights(output_path / f"{episode}_a2c.h5")
        self.save_weights(output_path / f"{episode}_a2c.keras")

    def load_model(self, output_path: Path) -> (int, int):
        """Load latest weights"""
        files = list(output_path.iterdir())
        episodes = []
        scores = []
        names = []
        if files:
            for file in files:
                if 'a2c' in file.name:
                    episode = int(file.name.split('_')[0])
                    score = int(file.name.split('_')[1])
                    episodes.append(episode)
                    scores.append(score)
                    names.append(f"{episode}_{score}")

            if episodes:
                # model_index = scores.index(max(scores))
                model_index = episodes.index(max(episodes))
                # latest_episode = max(episodes)
                print(f"Loaded weight for model {names[model_index]}!")
                self.load_weights(output_path / f"{names[model_index]}_a2c.keras")
                return max(episodes), max(scores)
        return 0, 0


# import tensorflow as tf
# from tensorflow.keras import layers
#
#
# class ActorCriticModel(tf.keras.Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCriticModel, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.dense1 = layers.Dense(100, activation="relu")
#         self.policy_logits = layers.Dense(action_size)
#         self.dense2 = layers.Dense(100, activation="relu")
#         self.values = layers.Dense(1)
#
#     def call(self, inputs):
#         # Forward pass
#         x = self.dense1(inputs)
#         logits = self.policy_logits(x)
#         v1 = self.dense2(inputs)
#         values = self.values(v1)
#         return logits, values
