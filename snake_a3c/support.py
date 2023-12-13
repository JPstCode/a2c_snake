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

    if episode % 100 == 0:
        print(
            f"Episode: {episode}, "
            f"loss: {episode_loss}, "
            f"reward: {episode_reward}, "
            f"episode_steps: {episode_steps}, "
            f"moving_average: {global_moving_average_reward}"
        )
    return 0
