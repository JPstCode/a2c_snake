import os
from collections import deque
from random import choice
import collections
import tqdm
import statistics
from pathlib import Path

from typing import Any, List, Sequence, Tuple
from model.actor_critic import ActorCritic
import tensorflow as tf
import numpy as np

from game_env import CNNGame as Game
from snake import rl_direction_map

tf.config.run_functions_eagerly(True)

output_path = Path(r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a2c")

env = Game(
    frame_x_size=50,
    frame_y_size=50,
    show_game=False,
    long_snake=True,
    block_size=10
)

eps = np.finfo(np.float32).eps.item()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = ActorCritic(state_size=env.get_observation().shape, action_size=4)
model(tf.expand_dims(env.get_observation(), 0))
model.load_model(output_path=output_path)


@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    env.snake.update_direction(rl_direction_map[action.item()])
    state, reward, done = env.update_game_rl()
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def run_episode(
    initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = env_step(action)

        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

    return returns


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int,
) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

        # Calculate the expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]
        ]

        # Calculate the loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500

# `CartPole-v1` is considered solved if average reward is >= 475 over 500
# consecutive trials
reward_threshold = 475
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    initial_state = env.reset_game()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

    # Show the average episode reward every 10 episodes
    # if i % 10 == 0:
    #     pass  # print(f'Episode {i}: average reward: {avg_reward}')

    if i % 1000 == 0:
        model.save_model(output_path=output_path, episode=i)

    # if running_reward > reward_threshold and i >= min_episodes_criterion:
    #     break
