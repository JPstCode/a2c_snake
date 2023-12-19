""""""

import threading
import os
from queue import Queue
from collections import deque
from pathlib import Path

import gym
import tensorflow as tf
import numpy as np

from a2c import ActorCriticModel
from memory import Memory
import parameters
import support
from game_env import CNNGame as Game
from snake import rl_direction_map

from matplotlib import pyplot as plt

tf.config.run_functions_eagerly(True)
eps = np.finfo(np.float32).eps.item()

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# global_best_score = 0

class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_reward_container = deque(maxlen=10000)
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(
        self,
        state_size,
        action_size,
        global_model,
        opt,
        result_queue,
        idx,
        game_name="CartPole-v0",
        save_dir="/tmp",
        best_score: int = 0,
        global_episode: int = 0
    ):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.worker_idx = idx
        self.game_name = game_name
        self.env = Game(
            frame_x_size=50,
            frame_y_size=50,
            show_game=False,
            long_snake=True,
            block_size=10,
            step_limit=25
        )
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.local_model(tf.expand_dims(self.env.get_observation(), axis=0))
        self.local_model.load_model(Path(r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a3c"))
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.best_score = best_score
        self.global_episode = global_episode

    def run(self):
        total_step = 1
        mem = Memory()
        time_count = 0
        while Worker.global_episode < parameters.MAX_EPS:
            current_state = self.env.reset_game()
            mem.clear()
            ep_reward = 0.0
            ep_steps = 0
            self.ep_loss = 0
            if Worker.best_score == 0:
                Worker.best_score = self.best_score
            if Worker.global_episode == 0:
                Worker.global_episode = self.global_episode

            # time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(np.expand_dims(current_state, axis=0))
                probs = tf.nn.softmax(logits)

                try:
                    action = np.random.choice(self.action_size, p=probs.numpy()[0])
                except ValueError:
                    print("JAAH")
                    action = np.random.choice(self.action_size)

                self.env.snake.update_direction(rl_direction_map[action])
                reward, done = self.env.update_game_rl()
                new_state = self.env.get_observation()

                # if done:
                #     reward = -10

                ep_reward += reward
                Worker.global_reward_container.append(ep_reward)

                mem.store(current_state, action, reward)

                # if time_count == parameters.UPDATE_FREQ:
                # if time_count == parameters.UPDATE_FREQ or done:
                if done: #and self.env.step > 10:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    # if time_count >= parameters.UPDATE_FREQ:
                        # print(f"UPDATE {self.worker_idx}")
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(
                            done, new_state, mem, parameters.GAMMA
                        )
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(
                        total_loss, self.local_model.trainable_variables
                    )
                    try:
                        # Push local gradients to global model
                        self.opt.apply_gradients(
                            zip(grads, self.global_model.trainable_weights)
                        )
                    except IndexError:
                        print()
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    # print()
                    if done:  # done and print information
                        Worker.global_moving_average_reward = np.mean(Worker.global_reward_container)
                        # if self.env.score > 0:
                        #     print()

                        support.record(
                            episode=Worker.global_episode,
                            episode_reward=ep_reward,
                            worker_idx=self.worker_idx,
                            global_moving_average_reward=Worker.global_moving_average_reward,
                            result_que=self.result_queue,
                            episode_loss=self.ep_loss,
                            episode_steps=self.env.step,
                            score=self.env.score
                        )

                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                        # if Worker.global_episode % 200 == 0:
                            with Worker.save_lock:
                                Worker.best_score = ep_reward
                                print(
                                    "Saving best model to {}, "
                                    "episode score: {} "
                                    "Worker idx {}".format(self.save_dir, ep_reward, self.worker_idx)
                                )
                                self.global_model.save_weights(
                                    os.path.join(
                                        self.save_dir,
                                        "{}_{}_a2c.keras".format(int(Worker.global_episode), int(Worker.best_score)),
                                        # "Snake_a2c.keras",
                                    )
                                )

                        Worker.global_episode += 1

                    mem.clear()
                    time_count = 0

                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1

        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.0  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)
            )[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for idx, reward in enumerate(memory.rewards):  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)

        # discounted_rewards.reverse()
        discounted_rewards = np.asarray(discounted_rewards, dtype=np.float32).reshape((len(discounted_rewards), 1))
        discounted_rewards = (discounted_rewards - tf.math.reduce_mean(discounted_rewards)) / (tf.math.reduce_std(discounted_rewards) + eps)

        action_logits_t = []
        values = []
        for state in memory.states:
            a, v = self.local_model(np.expand_dims(state, axis=0))
            action_logits_t.append(a[0])
            values.append(v[0])

        action_logits_t = tf.convert_to_tensor(action_logits_t)
        values = tf.convert_to_tensor(values)

        # action_logits_t, values = self.local_model(np.vstack(tf.expand_dims(memory.states, axis=0)))
        advantage = discounted_rewards - values

        # # action_log_probs = tf.math.log(action_logits_t)
        action_probs_t = tf.nn.softmax(action_logits_t)
        action_probs = []
        for idx, action in enumerate(memory.actions):
            action_probs.append(tf.expand_dims(action_probs_t[idx][action], axis=0))

        # action_probs = np.asarray(action_probs).reshape(len(action_probs), 1)
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        # actor_loss = -tf.math.reduce_sum(action_probs * advantage)

        critic_loss = huber_loss(values, discounted_rewards)

        total_loss = actor_loss + critic_loss
        return total_loss


        # # Get discounted rewards
        # discounted_rewards = []
        # for idx, reward in enumerate(memory.rewards):  # reverse buffer r
        #     reward_sum = reward + gamma * reward_sum
        #     discounted_rewards.append(reward_sum)
        #
        # discounted_rewards = np.asarray(discounted_rewards, dtype=np.float32).reshape((len(discounted_rewards), 1))
        # discounted_rewards = (discounted_rewards - tf.math.reduce_mean(discounted_rewards)) / (tf.math.reduce_std(discounted_rewards) + eps)
        #
        # action_logits_t, values = self.local_model(np.vstack(tf.expand_dims(memory.states, axis=0)))
        # # Get our advantages
        # advantage = (
        #     tf.convert_to_tensor(
        #         np.array(discounted_rewards)[:, None], dtype=tf.float32
        #     )
        #     - values
        # )
        #
        # # # Value loss
        # value_loss = advantage**2
        #
        # # Calculate our policy loss
        # actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)
        #
        # policy = tf.nn.softmax(action_logits_t)
        # entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)
        #
        # policy_loss = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=actions_one_hot, logits=action_logits_t
        # )
        # policy_loss *= tf.stop_gradient(advantage)
        # policy_loss -= 0.01 * entropy
        # total_loss2 = tf.reduce_mean((0.5 * value_loss + policy_loss))
        # return total_loss

    def play(self):
        env = gym.make(self.game_name, render_mode='human').unwrapped
        state = env.reset()[0]
        model = self.global_model
        model_path = os.path.join(self.save_dir, "model_{}.h5".format(self.game_name))
        model(tf.expand_dims(state, axis=0))
        print("Loading model from: {}".format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                jaa = env.render()
                policy, value = model(
                    tf.expand_dims(state, axis=0)
                )
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _, _ = env.step(action)
                reward_sum += reward
                print(
                    "{}. Reward: {}, action: {}".format(
                        step_counter, reward_sum, action
                    )
                )
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

    def play_snake(self):
        """"""
        weights_dir = Path(r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a2c")
        env = Game(
            frame_x_size=50,
            frame_y_size=50,
            show_game=True,
            long_snake=True,
            block_size=10
        )

        model = ActorCritic(state_size=env.get_observation().shape, action_size=4)
        model(tf.expand_dims(env.get_observation(), 0))
        model.load_model(output_path=weights_dir)

        n_episodes = 10000
        state = env.reset_game()
        state = tf.constant(state, dtype=tf.float32)
        for e in range(n_episodes):

            for i in range(2000):
                action_logits_t, value = model(tf.expand_dims(state, 0))
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
                action_probs_t = tf.nn.softmax(action_logits_t)
                # print(rl_direction_map[action.numpy().item()])
                env.snake.update_direction(rl_direction_map[action.numpy().item()])
                state, reward, done = env.update_game_rl()
                state = state.astype(np.float32)

                if done:
                    print(env.score)
                    state = env.reset_game()
                    state = tf.constant(state, dtype=tf.float32)

            print(env.score)
            state = env.reset_game()
            state = tf.constant(state, dtype=tf.float32)


if __name__ == '__main__':

    game_name = "CartPole-v0"
    env = gym.make(game_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    res_queue = Queue()
    idx = 0
    save_dir = r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a3c"
    global_model = ActorCriticModel(
        state_size, action_size
    )

    worker = Worker(
                state_size=state_size,
                action_size=action_size,
                global_model=global_model,
                opt=optimizer,
                result_queue=res_queue,
                idx=idx,
                game_name=game_name,
                save_dir=save_dir,
            )

    worker.play()