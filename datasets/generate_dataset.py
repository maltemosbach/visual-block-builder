from datasets import io
from examples.parser import VBBParser
import gym
import numpy as np
import os
import tqdm
import visual_block_builder


DATASET_KEYS = ['image', 'action', 'reward']


# Parse environment arguments.
parser = VBBParser()
parser.add_argument('--dataset_dir', type=str, default="./datasets/random_dataset",
                    help="Path to save the dataset.")
parser.add_argument('--num_episodes', type=int, default=10,
                    help="Number of episodes to collect for the dataset.")
parser.add_argument('--policy', type=str, choices=["random"], default="random", 
                    help="The behavior policy used to collect the dataset.")
parser.add_argument('--save_format', type=str, choices=['npz', 'png'], default='npz',
                    help="Format to save the dataset in.")
args = parser.parse_args()

env = gym.make(f'VisualBlockBuilder_{args.num_blocks}Blocks_SparseReward_DictstateObs_{args.stack_only}Stackonly_{args.case}Case_{args.viewpoint.title()}Viewpoint{args.robot.title()}Robot-v1')
os.makedirs(args.dataset_dir, exist_ok=True)
io.save_metadata(vars(args) , args.dataset_dir)
pbar = tqdm.tqdm(range(args.num_episodes))
pbar.set_description(f"Collecting dataset with {args.policy} policy")

for episode in pbar:
    episode_data = {key: [] for key in DATASET_KEYS}
    obs, done = env.reset(), False
    image = env.render(mode='rgb_array')
    episode_data['image'].append(image)
    while not done:
        if args.policy == "random":
            action = env.action_space.sample()
        else:
            raise NotImplementedError
        obs, reward, done, info = env.step(action)
        image = env.render(mode='rgb_array')
        episode_data['image'].append(image)
        episode_data['action'].append(action)
        episode_data['reward'].append(reward)
    getattr(io, f"save_{args.save_format}")(episode_data, episode, args.dataset_dir)
