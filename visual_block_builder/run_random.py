import gym
import numpy as np
import visual_block_builder
import matplotlib.pyplot as plt



class MatplotlibViewer:
    def __init__(self, title: str = "Viewer") -> None:
        self._imshow_initialized = False
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        plt.axis('off')
        plt.show(block=False)

    def _initialize_imshow(self, image: np.array) -> None:
        self.imshow = self.ax.imshow(image)
        self._imshow_initialized = True

    def _update_imshow(self, image: np.array) -> None:
        self.imshow.set_data(image)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def update(self, image: np.array) -> None:
        if self._imshow_initialized:
            self._update_imshow(image)
        else:
            self._initialize_imshow(image)


env = gym.make('VisualBlockBuilder_3Blocks_SparseReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1')
viewer = MatplotlibViewer()


while True:
    obs, done = env.reset(), False

    while not done:
        viewer.update(env.render(mode='rgb_array'))
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("obs: ", obs)
