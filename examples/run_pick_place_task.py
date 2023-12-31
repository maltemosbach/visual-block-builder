from examples.gui import GUI
import gym
from examples.parser import ReachPickPlaceTargetParser
from tabulate import tabulate
import tqdm
import visual_block_builder


# Parse environment arguments.
parser = ReachPickPlaceTargetParser()
parser.add_argument("--policy", type=str, choices=["random", "user"], default="random",
                    help="The policy used to control the robot. 'user' for interactive keyboard control, and 'random' for a random policy.")
args = parser.parse_args()

# Create environment.
env = gym.make(f'PickAndPlace{args.case}Block_{args.num_distractors + 1}Blocks_{args.target_size.title()}Target_{args.object_size.title()}Objects_SparseReward_DictstateObs_{args.viewpoint.title()}Viewpoint{args.robot.title()}Robot-v1')
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
