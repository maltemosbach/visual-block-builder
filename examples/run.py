import argparse
from examples.gui import GUI
import gym
import numpy as np
from tabulate import tabulate
import tqdm
import visual_block_builder


parser = argparse.ArgumentParser()
parser.add_argument("--case", type=str, choices=["Singletower", "Pyramid", "Multitower", "All"], default="Singletower",
                    help="The case to run. Must be one of 'Singletower', 'Pyramid', 'Multitower', or 'All'.")
parser.add_argument("--num_blocks", type=int, default=2,
                    help="The number of blocks in the environment. Must be between 1 and 17.")
parser.add_argument("--policy", type=str, choices=["random", "user"], default="random",
                    help="The policy used to control the robot. 'user' for interactive keyboard control, and 'random' for a random policy.")
parser.add_argument("--stack_only", action="store_true")
parser.add_argument("--viewpoint", type=str, choices=["frontview", "topview", "external_camera_0"], default="frontview")
args = parser.parse_args()


env = gym.make(f'VisualBlockBuilder_{args.num_blocks}Blocks_SparseReward_DictstateObs_{args.stack_only}Stackonly_{args.case}Case_{args.viewpoint.title()}Viewpoint-v1')
env.render(mode='rgb_array')
gui = GUI(fps=env.metadata['video.frames_per_second'])

if args.policy == "user":
    print("Running GUI:")
    print(tabulate([['W', 'forward'], ['S', 'backward'], ['A', 'left'], ['D', 'right'], ['shift', 'down'], ['space', 'up'], ['E', 'open gripper'], ['Q', 'close gripper']], headers=['Key', 'Action']) + "\n")
elif args.policy == "random":
    print("Running random policy:")

episode = 0
while True:
    obs, done = env.reset(), False
    episode += 1
    pbar = tqdm.tqdm(range(env._max_episode_steps))
    pbar.set_description(f"Episode {episode}")
    for step in pbar:
        gui.display(env.render(mode='rgb_array'))
        if args.policy == "random":
            action = env.action_space.sample()
        elif args.policy == "user":
            action = gui.get_action()
        obs, reward, done, info = env.step(action)
        pbar.set_postfix(reward=reward)
        if done:
            break
