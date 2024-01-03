import random
import time
from copy import deepcopy
from typing import Optional
from collections import deque
import sys
import json

import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt

from snake import Snake, rl_direction_map


# Colors (R, G, B)
black = (0, 0, 0)
white = (255, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
orange = (0, 191, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.3
fontColor = (255, 255, 255)
thickness = 1
lineType = 2

class CNNGame:
    def __init__(
            self, frame_x_size: int,
            frame_y_size: int,
            block_size: int = 5,
            show_game: bool = False,
            long_snake: bool = False,
            step_limit: int = 25
    ):

        # Visualization
        self.frame_x_size: int = frame_x_size
        self.frame_y_size: int = frame_y_size
        self.block_size: int = block_size
        self.canvas = np.zeros(
            (self.frame_y_size + block_size, self.frame_x_size + block_size, 3), dtype=np.uint8
        )
        self.current_canvas = self.canvas

        # Snake
        self.snake: Optional[Snake] = None
        # Food
        self.food_position: list = [0, 0]
        self.dist_to_food = 0

        # self.difficulty = difficulty
        self.game_lost = False
        self.show_game = show_game
        self.long_snake = long_snake

        self.observation_que: deque = deque(maxlen=4)
        self.observation: Optional[npt.NDArray] = None

        self.prev_dist_to_food: Optional[int] = None
        self.closer = False
        self.episode: Optional[int] = None
        self.score: int = 0
        self.step: int = 0
        self.eaten_step: int = 0
        self.step_limit: int = step_limit

        self.reset_game()

    def reset_game(self) -> np.ndarray:
        self.snake = initialize_snake(block_size=self.block_size)
        self.food_position, _ = self.position_food()
        self.dist_to_food = self.get_dist_to_food()
        self.draw_elements()
        self.observation_que = deque(maxlen=4)
        observation = self.get_observation()
        self.step = 0
        self.eaten_step = 0
        self.score = 0
        self.game_lost = False

        return observation

    def get_dist_to_food(self):
        return np.sqrt(
            (self.snake.head_position[0] - self.food_position[0]) ** 2 +
            (self.snake.head_position[1] - self.food_position[1]) ** 2
        )

    def draw_elements(self):
        canvas = self.canvas.copy()

        # Draw snake body
        for idx, pos in enumerate(self.snake.body):
            if idx == 0:
                cv2.rectangle(
                    canvas, (pos[0] + 1, pos[1] + 1), (pos[0] + self.block_size - 1, pos[1] + self.block_size - 1), white, -1
                    # canvas, (pos[0], pos[1]), (pos[0] + self.block_size, pos[1] + self.block_size), white, -1
                )
            else:
                cv2.rectangle(
                    canvas, (pos[0] + 1, pos[1] + 1), (pos[0] + self.block_size - 1, pos[1] + self.block_size - 1), green, -1
                    # canvas, (pos[0], pos[1]), (pos[0] + self.block_size, pos[1] + self.block_size), green, -1
                )

        if not self.snake.eaten:
            food_color = red
        else:
            food_color = (0, 200, 255)

        cv2.rectangle(
            canvas,
            (self.food_position[0] + 1, self.food_position[1] + 1),
            (self.food_position[0] + self.block_size - 1, self.food_position[1] + self.block_size - 1),
            food_color,
            -1,
        )

        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        self.current_canvas = canvas

        if self.show_game:
            cv2.imshow("window", canvas)
            cv2.waitKey(5)
            time.sleep(1 / 100)
        return

    def update_game_rl(self):

        self.snake.move(self.snake.direction)
        self.snake.grow(self.food_position)

        self.draw_elements()

        reward = 0.0

        # self.game_lost = self.check_if_lost()
        #
        # if self.game_lost:
        #     reward = 0
        if self.check_if_lost():
            self.game_lost = True
            reward = -1

        # if self.step > self.step_limit:
        if self.eaten_step > self.step_limit:
            self.game_lost = True
#            reward = 0

        self.step += 1
        self.eaten_step += 1

        if self.snake.eaten:
            self.score += 1
            self.snake.eaten = False
            self.food_position, done = self.position_food()
            self.game_lost = done
            reward = 1
            self.eaten_step = 0

        # return self.get_observation(), reward, self.game_lost
        return reward, self.game_lost

    def position_food(self) -> (list[int, int], bool):

        # TODO check if won game
        free_positions_x = np.arange(0, self.frame_x_size + 10, 10)
        free_positions_y = np.arange(0, self.frame_y_size + 10, 10)

        free_positions = []
        for ix in free_positions_x:
            for iy in free_positions_y:
                pos = [ix, iy]
                if pos == self.food_position:
                    continue

                if pos in self.snake.body:
                    continue

                free_positions.append([ix, iy])

        if len(free_positions) == 0:
            print("WIN!!!!!")
            return [], True

        food_position = random.choice(free_positions)
        return food_position, False

    def check_if_lost(self):

        # Out of frame X
        if (
            self.snake.head_position[0] < 0
            or self.snake.head_position[0] > (self.frame_x_size)
        ):
            # for i in range(1, 5):
            #     plt.subplot(1,4,i)
            #     plt.imshow(self.observation_que[i - 1])
            # print(self.snake.direction)
            # plt.show()

            # print("Out of frame")
            return True

        # Out of frame Y
        if (
            self.snake.head_position[1] < 0
            or self.snake.head_position[1] > (self.frame_y_size)
        ):
            # for i in range(1, 5):
            #     plt.subplot(1,4,i)
            #     plt.imshow(self.observation_que[i - 1])
            # print(self.snake.direction)
            # plt.show()

            # print("Out of frame")
            return True

        # for block in snake.body[1:]:
        if self.snake.head_position in self.snake.body[1:]:
            # print("Collision to body")
            # for i in range(1, 5):
            #     plt.subplot(1,4,i)
            #     plt.imshow(self.observation_que[i - 1])
            # print(self.snake.direction)
            # plt.show()

            return True

        return False

    def get_observation(self, epsilon: Optional[float] = None):
        """"""
        # Fill observation que
        if len(self.observation_que) == 0:
            for i in range(self.observation_que.maxlen):
                self.update_game_rl()
                self.observation_que.append(self.current_canvas / 255)

        else:
            self.observation_que.append(self.current_canvas / 255)

        return np.asarray(list(self.observation_que))

def initialize_snake(
        start_x: int = 0,
        start_y: int = 30,
        block_size: int = 5,
        start_direction: str = "RIGHT",
) -> Snake:
    """Initialize snake object"""

    #with open(r"C:\tmp\a2c-long\snake.json") as f:
    #    body = json.load(f)

    return Snake(
        head_position=[start_x, start_y],
        #direction="DOWN",
        #body=body,
        direction=start_direction,
        body=[
             [start_x, start_y],
             [start_x, start_y - block_size],
             [start_x, start_y - (2 * block_size)]
        ],
        block_size=block_size
    )
