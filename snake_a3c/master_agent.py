""""""
from pathlib import Path
import os
from queue import Queue
import multiprocessing

import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from snake_a3c.a2c import ActorCriticModel
from snake_a3c.random_agent import RandomAgent
from snake_a3c import parameters
from snake_a3c.worker import Worker
from snake_a3c.game_env import CNNGame as Game

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class MasterAgent:
    def __init__(self):
        self.game_name = "Snake"
        save_dir = r"C:\tmp\a2c"
        #save_dir = r"C:\tmp\a2c-long"
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # env = gym.make(self.game_name)
        env = Game(
            frame_x_size=50,
            frame_y_size=50,
            show_game=False,
            long_snake=True,
            block_size=10,
            step_limit=50
        )
        self.state_size = env.get_observation().shape
        self.action_size = 4
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001) #, use_locking=True)
        # print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(
            self.state_size, self.action_size
        )  # global network
        self.global_model(tf.expand_dims(env.get_observation(), axis=0))
        self.latest_episode, self.best_score = self.global_model.load_model(output_path=Path(save_dir))

    def train(self):
        if parameters.ALGORITHM == "random":
            random_agent = RandomAgent(self.game_name, max_eps=parameters.MAX_EPS)
            random_agent.run()
            return

        res_queue = Queue()

        workers = [
            Worker(
                self.state_size,
                self.action_size,
                self.global_model,
                self.opt,
                res_queue,
                i,
                game_name=self.game_name,
                save_dir=self.save_dir,
                best_score=self.best_score,
                global_episode=self.latest_episode
            )
            # for i in range(multiprocessing.cpu_count())
            # for i in range(10)
            # for i in range(8)
            # for i in range(5)
            # for i in range(3)
            for i in range(2)
            # for i in range(1)
        ]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel("Moving average ep reward")
        plt.xlabel("Step")
        plt.savefig(
            os.path.join(self.save_dir, "{} Moving Average.png".format(self.game_name))
        )
        plt.show()

if __name__ == '__main__':

    master_agent = MasterAgent()
    master_agent.train()
    print()