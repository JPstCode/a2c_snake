""""""
import os
from queue import Queue
import multiprocessing

import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from a2c import ActorCriticModel
from random_agent import RandomAgent
import parameters
from worker import Worker


class MasterAgent:
    def __init__(self):
        self.game_name = "CartPole-v0"
        save_dir = r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a3c"
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01) #, use_locking=True)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(
            self.state_size, self.action_size
        )  # global network
        self.global_model(
            tf.convert_to_tensor(
                np.random.random((1, self.state_size)), dtype=tf.float32
            )
        )

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
            )
            for i in range(multiprocessing.cpu_count())
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