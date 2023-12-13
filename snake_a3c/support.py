""""""
from __future__ import annotations
from queue import Queue


def record(
        episode: int,
        episode_reward: float,
        worker_idx: int,
        global_moving_average_reward: int,
        result_que: Queue,
        episode_loss: int,
        episode_steps: int
) -> int:
    """"""

    print("Do recording")
    return 0
