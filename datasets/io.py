import cv2
import json
import numpy as np
import os


def save_metadata(args: dict, dataset_dir: str) -> None:
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(args, f)


def save_npz(episode_data: dict, episode: int, dataset_dir: str) -> None:
    for key in episode_data.keys():
        episode_data[key] = np.array(episode_data[key])
    np.savez(os.path.join(dataset_dir, f"episode_{episode}.npz"), **episode_data)


def save_png(episode_data: dict, episode: int, dataset_dir: str) -> None:
    os.makedirs(os.path.join(dataset_dir, f"episode_{episode}"), exist_ok=True)
    time_step = 0
    for image in episode_data['image']:
        image = image[..., ::-1].astype(np.uint8)
        cv2.imwrite(os.path.join(dataset_dir, f"episode_{episode}", f"image_{time_step}.png"), image)
        time_step += 1
    for key in episode_data.keys():
        if key != 'image':
            np.save(os.path.join(dataset_dir, f"episode_{episode}", f"{key}.npy"), np.array(episode_data[key]))
