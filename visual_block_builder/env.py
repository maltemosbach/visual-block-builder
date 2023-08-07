import numpy as np
from fetch_block_construction.envs.robotics.fetch.construction import FetchBlockConstructionEnv
from fetch_block_construction.envs.robotics import fetch_env
from gym import utils as gym_utils
import os
import pkg_resources
import tempfile
from typing import Dict, Any
from visual_block_builder.assets.generate_multi_camera_xml import generate_multi_camera_xml


class VisualBlockBuilderEnv(FetchBlockConstructionEnv):
    def __init__(self, initial_qpos: Dict[str, Any], num_blocks: int = 1, reward_type: str = "incremental",
                    obs_type: str = "np", stack_only: bool = False, case: str = "Singletower", viewpoint: str = "topview", 
                    robot: str = "default", width: int = 1024, height: int = 1024) -> None:
        self.num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.stack_only = stack_only
        self.case = case
        self.viewpoint = viewpoint
        self.width = width
        self.height = height


        # Ensure we get the path separator correct on windows
        # MODEL_XML_PATH = os.path.join('fetch', F'stack{self.num_blocks}.xml')

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_blocks, robot))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, reward_type, obs_type, 0, stack_only)
        self.render_image_obs = False

    def render(self, mode='human', size=None):
        self._render_callback()
        data = self.sim.render(self.width, self.height, camera_name=self.viewpoint)
        # original image is upside-down, so flip it
        return data[::-1, :, :]


class ReachTargetEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, initial_qpos: Dict[str, Any], num_distractors: int = 1, reward_type: str = "sparse",
                 obs_type: str = "np", case: str = "Specific", viewpoint: str = "topview", robot: str = "default", width: int = 1024,
                 height: int = 1024) -> None:
        self.num_distractors = num_distractors
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.case = case

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_distractors, robot, task='reach'))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0,
            obj_range=0, target_range=0, distance_threshold=0.035,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)

        os.remove(MODEL_XML_PATH)

    def render(self, mode='human', size=None):
        self._render_callback()
        data = self.sim.render(self.width, self.height, camera_name=self.viewpoint)
        # original image is upside-down, so flip it
        return data[::-1, :, :]

    def sample_pos_on_table(self, object_size):
        return np.concatenate([self.np_random.uniform(-self.table_size[0] + object_size,
                                                      self.table_size[0] - object_size - 0.1, size=1) + self.x_offset,
                               self.np_random.uniform(-self.table_size[1] + object_size,
                                                      self.table_size[1] - object_size, size=1) + self.y_offset,
                               np.array([self.height_offset + self.table_size[2] + object_size])])

    def _sample_goal(self):
        goal = self.sample_pos_on_table(self.target_size)
        goal += self.target_offset
        return goal.copy()

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.x_offset = self.sim.data.get_body_xpos('table0')[0]
        self.y_offset = self.sim.data.get_body_xpos('table0')[1]
        self.height_offset = self.sim.data.get_body_xpos('table0')[2]

        self.target_size = self.sim.model.site_size[self.sim.model.site_name2id('target0')][0]
        self.table_size = self.sim.model.geom_size[self.sim.model.geom_name2id('table0')]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize color of target and distractors
        if self.case == 'Distinct':
            target_color = np.random.randint(0, 256, size=3)
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = target_color / 255

            same_color = True
            while same_color:
                distractor_color = np.random.randint(0, 256, size=3)
                same_color = np.array_equal(target_color, distractor_color)

            for i in range(self.num_distractors):
                self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = distractor_color / 255

        # Randomize start position of distractors.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        ball_positions = [self.goal]
        for i in range(self.num_distractors):
            pos_not_valid = True
            while pos_not_valid:
                distractor_xpos = self.sample_pos_on_table(self.target_size)

                pos_not_valid = False
                for position in ball_positions:
                    pos_not_valid = np.linalg.norm(distractor_xpos - position) < 2 * self.target_size
                    if pos_not_valid:
                        break

            ball_positions.append(distractor_xpos)

            site_id = self.sim.model.site_name2id(f'distractor{i}')
            self.sim.model.site_pos[site_id] = distractor_xpos - sites_offset[0]

        self.sim.forward()
        return True

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self.goal = self._sample_goal().copy()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        return obs

    def check_distractor_dist(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')

        for i in range(self.num_distractors):
            if np.linalg.norm(
                    grip_pos.copy() - self.sim.data.get_site_xpos(f'distractor{i}').copy()) <= self.distance_threshold:
                return -np.inf

        return 0

    def compute_reward_image(self):
        distractor_dist = self.check_distractor_dist()
        return super().compute_reward_image() + distractor_dist

    def compute_reward(self, achieved_goal, goal, info):
        distractor_dist = self.check_distractor_dist()
        return super().compute_reward(achieved_goal, goal, info) + distractor_dist


class PickAndPlaceBlockEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, initial_qpos: Dict[str, Any], reward_type='sparse', obs_type='np', render_size=42,
                 num_blocks: int = 1, case: str = "Specific", viewpoint: str = "topview", robot: str = "default",
                 width: int = 1024, height: int = 1024):

        self.num_blocks = num_blocks
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.case = case
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_blocks, robot, task='pick_place'))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)
        gym_utils.EzPickle.__init__(self, reward_type, obs_type, render_size)

        os.remove(MODEL_XML_PATH)

    def render(self, mode='human', size=None):
        self._render_callback()
        data = self.sim.render(self.width, self.height, camera_name=self.viewpoint)
        # original image is upside-down, so flip it
        return data[::-1, :, :]

    def _reset_sim(self):
        assert self.num_blocks <= 17  # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize color of objects
        if self.case == 'Distinct':
            target_object_color = np.random.randint(0, 256, size=3)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[1])][:3] = target_object_color / 255

            same_color = True
            while same_color:
                distractor_object_color = np.random.randint(0, 256, size=3)
                same_color = np.array_equal(target_object_color, distractor_object_color)

            for object_name in self.object_names[1:]:
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = distractor_object_color / 255

        # Randomize start position of objects.

        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                  size=2)

            while not ((np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1) and np.all(
                    [np.linalg.norm(object_xypos - other_xpos) >= 0.06 for other_xpos in prev_obj_xpos])):
                object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                      size=2)

            prev_obj_xpos.append(object_xypos)

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.3)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()
