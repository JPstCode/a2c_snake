""""""
from pathlib import Path

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from snake_a3c.game_env import CNNGame as Game
from snake_a3c.a2c import ActorCriticModel as ActorCritic
from snake_a3c.snake import rl_direction_map

weights_dir = Path(r"C:\tmp\a2c")
#weights_dir = Path(r"C:\tmp\a2c-140000")
env = Game(
    frame_x_size=50,
    frame_y_size=50,
    show_game=True,
    long_snake=True,
    block_size=10,
    step_limit=100
)

model = ActorCritic(state_size=env.get_observation().shape, action_size=4)
model(tf.expand_dims(env.get_observation(), 0))
model.load_model(output_path=weights_dir)

while True:
    state = env.reset_game()
    states = []
    rewards = []
    actions = []

    for i in range(2000):
        action_logits, value = model(tf.expand_dims(state, 0))
        probs = tf.nn.softmax(action_logits)
        action = np.random.choice(4, p=probs.numpy()[0])

        env.snake.update_direction(rl_direction_map[action])
        reward, done = env.update_game_rl()

        new_state = env.get_observation()

        states.append(state)
        rewards.append(reward)
        actions.append(action)
        # print("Steps: ", env.eaten_step)
        # if reward != 0:
        # #if env.score > 10:
        #     print("R, D, L, U")
        #     print(value)
        #     print(tf.nn.softmax(action_logits))
        #     print(rl_direction_map[action])
        #     print(rewards[-1])
        #     plt.figure()
        #     for i in range(4):
        #         plt.subplot(1, 4, i + 1)
        #         plt.imshow(states[-1][i])
        #
        #     plt.show()
        #     print('----')
        #
        # #
        # #     print()

        if reward > 0:
            print()

        if done:
            print(env.score)
            break

        state = new_state




