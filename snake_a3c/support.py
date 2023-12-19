""""""
from __future__ import annotations
from pathlib import Path
from queue import Queue
import os


output_path = Path(r"C:\Users\juhop\Documents\Projects\ML\Snake-AI-models\a3c")
def check_log_file(worker_idx: int) -> Path:
    """"""
    filename = f"worker_{worker_idx}.txt"
    files = os.listdir(output_path)
    if filename not in files:
        with open(output_path / filename, 'w') as file:
            # You can write initial content to the file if needed
            file.write(f"episode, loss, reward, steps, moving_average, score \n")
    return output_path / filename


    # log_files = []
    # for file in output_path.iterdir():
    #     if file.suffix != '.txt':
    #         continue
    #     log_files.append(file)
    #

    #
    # print()



def record(
        episode: int,
        episode_reward: float,
        worker_idx: int,
        global_moving_average_reward: int,
        result_que: Queue,
        episode_loss: int,
        episode_steps: int,
        score: int
) -> None:
    """"""

    log_filepath = check_log_file(worker_idx=worker_idx)
    with open(log_filepath, 'a') as file:
        # You can write initial content to the file if needed
        file.write(f"{episode}, {episode_loss}, {episode_reward}, {episode_steps}, {global_moving_average_reward}, {score} \n")

    if episode % 50 == 0:

        print(
            f"Episode: {episode}, "
            f"loss: {episode_loss}, "
            f"reward: {episode_reward}, "
            f"episode_steps: {episode_steps}, "
            f"moving_average: {global_moving_average_reward}, "
            f"Worker idx: {worker_idx}, "
            f"Score: {score}"
        )
