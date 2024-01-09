import numpy as np
from fetch_block_construction.envs.robotics.fetch.construction import FetchBlockConstructionEnv
from fetch_block_construction.envs.robotics import fetch_env
from gym import utils as gym_utils
import os
import pkg_resources
import tempfile
from typing import Dict, Any
from visual_block_builder.assets.generate_multi_camera_xml import generate_multi_camera_xml
from mujoco_py.generated import const


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
                 height: int = 1024, target_size="small") -> None:
        self.num_distractors = num_distractors
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.case = case
        self.target_size = {"small": 0.02, "medium": 0.03, "large": 0.04}[target_size]

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_distractors, robot, task='reach', target_size=self.target_size))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0,
            obj_range=0, target_range=0, distance_threshold=self.target_size + 0.015,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_distractors, reward_type, obs_type, case, viewpoint, robot, width, height, target_size)

        os.remove(MODEL_XML_PATH)

    def render(self, mode='human', size=None):
        self._render_callback()
        size = size if size is not None else (self.width, self.height)
        if mode == "seg":
            seg = self.sim.render(size[0], size[1], camera_name=self.viewpoint, segmentation=True)
            ids = seg[:, :, 1]

            # table to background
            id = self.sim.model.geom_name2id("table0")
            ids[ids == id] = -1

            for id in np.unique(ids):
                if id == -1:
                    continue

                name = self.sim.model.geom_id2name(id)
                if name is not None and ('gripper' in name or 'forearm' in name or 'wrist' in name or 'elbow' in name or
                                         'upperarm' in name):
                    ids[ids == id] = 254
                # robot to background
                elif name is not None and 'robot' in name:
                    ids[ids == id] = -1

            return (ids[::-1, :] + 1).astype(np.uint8)
        else:
            # original image is upside-down, so flip it
            return self.sim.render(size[0], size[1], camera_name=self.viewpoint)[::-1, :, :]

    def sample_pos(self, object_size):
        pos = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        pos[2] = self.height_offset + self.table_size[2] + object_size if pos[2] < self.height_offset + self.table_size[2] + object_size else pos[2]
        return pos.copy()

    def _sample_goal(self):
        return self.sample_pos(self.target_size)

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.height_offset = self.sim.data.get_body_xpos('table0')[2]
        self.table_size = self.sim.model.geom_size[self.sim.model.geom_name2id('table0')]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize color of target and distractors
        if self.case == 'Distinct':
            target_color = self.np_random.randint(0, 256, size=3)
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = target_color / 255

            color_dist = 0
            while color_dist < 100:
                distractor_color = self.np_random.randint(0, 256, size=3)
                color_dist = np.linalg.norm(target_color - distractor_color)

            for i in range(self.num_distractors):
                self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = distractor_color / 255
        # Randomize colors of distractors
        elif self.case == 'Specific':
           target_color = np.array([255, 0, 0])
           self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = target_color / 255

           for i in range(self.num_distractors):
               color_dist = 0
               while color_dist < 100:
                   distractor_color = self.np_random.randint(0, 256, size=3)
                   color_dist = np.linalg.norm(target_color - distractor_color)

               self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = distractor_color / 255
        else:
           target_color =  self.np_random.randint(0, 256, size=3)
           self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = target_color / 255

           for i in range(self.num_distractors):
               distractor_color = self.np_random.randint(0, 256, size=3)
               self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = distractor_color / 255


        # Randomize start position of distractors.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        ball_positions = [self.goal]
        for i in range(self.num_distractors):
            pos_not_valid = True
            while pos_not_valid:
                distractor_xpos = self.sample_pos(self.target_size)

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
                return -1

        return 0

    def compute_reward_image(self):
        distractor_dist = self.check_distractor_dist()
        return super().compute_reward_image()

    def compute_reward(self, achieved_goal, goal, info):
        distractor_dist = self.check_distractor_dist()
        return super().compute_reward(achieved_goal, goal, info)


class PickAndPlaceBlockEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, initial_qpos: Dict[str, Any], reward_type='sparse', obs_type='np', render_size=42,
                 num_blocks: int = 1, case: str = "Specific", viewpoint: str = "topview", robot: str = "default",
                 width: int = 1024, height: int = 1024, target_size="small", object_size="small"):

        self.num_blocks = num_blocks
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.case = case
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.target_size = {"small": 0.02, "medium": 0.03, "large": 0.04}[target_size]
        self.object_size = {"small": 0.025, "medium": 0.035, "large": 0.045}[object_size]

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_blocks, robot, task='pick_place', target_size=self.target_size, object_size=self.object_size))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=self.target_size + self.object_size,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)
        gym_utils.EzPickle.__init__(self, reward_type, obs_type, render_size)

        gym_utils.EzPickle.__init__(self, initial_qpos, reward_type, obs_type, render_size, num_blocks, case, viewpoint, robot, width, height, target_size, object_size)

        os.remove(MODEL_XML_PATH)

    def render(self, mode='human', size=None):
        self._render_callback()
        size = size if size is not None else (self.width, self.height)
        if mode == "seg":
            seg = self.sim.render(size[0], size[1], camera_name=self.viewpoint, segmentation=True)
            ids = seg[:, :, 1]

            # table to background
            id = self.sim.model.geom_name2id("table0")
            ids[ids == id] = -1

            for id in np.unique(ids):
                if id == -1:
                    continue

                name = self.sim.model.geom_id2name(id)
                if name is not None and ('forearm' in name or 'wrist' in name or 'elbow' in name or
                                         'upperarm' in name or 'gripper' in name):
                    ids[ids == id] = 254
                # robot to background
                elif name is not None and 'robot' in name:
                    ids[ids == id] = -1

            return (ids[::-1, :] + 1).astype(np.uint8)
        else:
            # original image is upside-down, so flip it
            return self.sim.render(size[0], size[1], camera_name=self.viewpoint)[::-1, :, :]

    def _reset_sim(self):
        assert self.num_blocks <= 17  # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize color of objects
        if self.case == 'Distinct':
            target_object_color = self.np_random.randint(0, 256, size=3)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = target_object_color / 255

            color_dist = 0
            while color_dist < 100:
                distractor_object_color = self.np_random.randint(0, 256, size=3)
                color_dist = np.linalg.norm(target_object_color - distractor_object_color)

            for object_name in self.object_names[1:]:
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = distractor_object_color / 255
        # Randomize colors of distractors
        elif self.case == 'Specific':
           target_object_color = np.array([0, 255, 0])
           self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = target_object_color / 255

           for i in range(self.num_blocks - 1):
               color_dist = 0
               while color_dist < 100:
                   distractor_object_color = self.np_random.randint(0, 256, size=3)
                   color_dist = np.linalg.norm(target_object_color - distractor_object_color)

               self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[i+1])][:3] = distractor_object_color / 255
        else:
           target_object_color = np.array([0, 255, 0])
           self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = target_object_color / 255

           for i in range(self.num_blocks):
               object_color = self.np_random.randint(0, 256, size=3)

               self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[i])][:3] = object_color / 255

        # Randomize start position of objects.

        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                  size=2)

            while not np.all([np.linalg.norm(object_xypos - other_xpos) >= 2 * np.sqrt(2) * self.object_size for other_xpos in prev_obj_xpos]):
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
        goal = self.sim.data.get_joint_qpos(F"{self.object_names[0]}:joint")[:3]
        object_positions = [self.sim.data.get_joint_qpos(F"{obj_name}:joint") for obj_name in self.object_names]
        while not (np.all([np.linalg.norm(goal - other_xpos[:3]) >= self.target_size + np.sqrt(2) * self.object_size for other_xpos in object_positions])):
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.3)
        return goal.copy()

    def compute_reward(self, achieved_goal, goal, info):
        grip_dist = np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy() - np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy())
        return max(super().compute_reward(achieved_goal, goal, info), -0.5) + 0.5 * max(-grip_dist, -0.5)


class PickAndPlaceSortEnv(FetchBlockConstructionEnv):
    def __init__(self, initial_qpos: Dict[str, Any], num_blocks: int = 1, reward_type: str = "incremental",
                    obs_type: str = "np", viewpoint: str = "topview", target_size="small", object_size="small",
                    robot: str = "default", width: int = 1024, height: int = 1024) -> None:
        self.num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.target_size = {"small": 0.02, "medium": 0.03, "large": 0.04}[target_size]
        self.object_size = {"small": 0.025, "medium": 0.035, "large": 0.045}[object_size]
        self.case = "Singletower"


        # Ensure we get the path separator correct on windows
        # MODEL_XML_PATH = os.path.join('fetch', F'stack{self.num_blocks}.xml')

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_blocks, robot, target_size=self.target_size, object_size=self.object_size))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=self.target_size + self.object_size,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, reward_type, obs_type, viewpoint, target_size, object_size, robot, width, height)

        self.render_image_obs = False

    def render(self, mode='human', size=None):
        self._render_callback()
        size = size if size is not None else (self.width, self.height)
        if mode == "seg":
            seg = self.sim.render(size[0], size[1], camera_name=self.viewpoint, segmentation=True)
            ids = seg[:, :, 1]

            # table to background
            id = self.sim.model.geom_name2id("table0")
            ids[ids == id] = -1

            for id in np.unique(ids):
                if id == -1:
                    continue

                name = self.sim.model.geom_id2name(id)
                if name is not None and ('forearm' in name or 'wrist' in name or 'elbow' in name or
                                         'upperarm' in name or 'gripper' in name or 'shoulder' in name):
                    ids[ids == id] = 254
                # robot to background
                elif name is not None and 'robot' in name:
                    ids[ids == id] = -1

            return (ids[::-1, :] + 1).astype(np.uint8)
        else:
            # original image is upside-down, so flip it
            return self.sim.render(size[0], size[1], camera_name=self.viewpoint)[::-1, :, :]

    def _sample_goal(self):
        goals = []

        goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal_object0 += self.target_offset
        goal_object0[2] = self.height_offset
        goals.append(goal_object0)

        for i in range(self.num_blocks - 1):
            goal_objecti = goals[0]
            while not (np.all([np.linalg.norm(goal_objecti - other_xpos[:3]) >= 3 * self.object_size for other_xpos in goals])):
                goal_objecti = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal_objecti += self.target_offset
                goal_objecti[2] = self.height_offset

            goals.append(goal_objecti)

        return np.concatenate(goals, axis=0).copy()

    def compute_reward(self, achieved_goal, goal, info):
        return max(-np.sum(self.subgoal_distances(achieved_goal[:-3], goal), axis=0), -0.5 * self.num_blocks)

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = self.subgoal_distances(achieved_goal[:-3], desired_goal)
        if np.sum([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances]) == 0:
            return True
        else:
            return False

    def _reset_sim(self):
        assert self.num_blocks <= 17  # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize color of objects
        object_0_color = self.np_random.randint(0, 256, size=3)
        self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = object_0_color / 255
        self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = object_0_color / 255
        self.sim.model.site_rgba[self.sim.model.site_name2id(f'target0')][3] = 1

        object_colors = [object_0_color]

        for i in range(self.num_blocks - 1):
            color_dist = 0
            while color_dist < 100:
                object_color = self.np_random.randint(0, 256, size=3)
                color_dist = min(np.linalg.norm(other_object_color - object_color) for other_object_color in object_colors)

            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[i + 1])][:3] = object_color / 255
            self.sim.model.site_rgba[self.sim.model.site_name2id(f'target{i+1}')][:3] = object_color / 255
            self.sim.model.site_rgba[self.sim.model.site_name2id(f'target{i+1}')][3] = 1
            object_colors.append(object_color)

        # Randomize start position of objects.
        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                  size=2)

            while not np.all([np.linalg.norm(object_xypos - other_xpos) >= 2 * np.sqrt(2) * self.object_size for other_xpos in prev_obj_xpos]):
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


class RenderEnv(FetchBlockConstructionEnv):
    def __init__(self, initial_qpos: Dict[str, Any], num_blocks: int = 0, num_targets=1, reward_type: str = "incremental",
                    obs_type: str = "np", viewpoint: str = "topview", target_size="small", object_size="small",
                    robot: str = "default", width: int = 1024, height: int = 1024) -> None:
        self.num_blocks = num_blocks
        self.num_targets = num_targets
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.viewpoint = viewpoint
        self.width = width
        self.height = height
        self.target_size = {"small": 0.02, "medium": 0.03, "large": 0.04}[target_size]
        self.object_size = {"small": 0.025, "medium": 0.035, "large": 0.045}[object_size]
        self.case = "Singletower"


        # Ensure we get the path separator correct on windows
        # MODEL_XML_PATH = os.path.join('fetch', F'stack{self.num_blocks}.xml')

        with tempfile.NamedTemporaryFile(mode='wt', dir=pkg_resources.resource_filename('fetch_block_construction', 'envs/robotics/assets/fetch'), delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(self.num_blocks, robot, task='render', target_size=self.target_size, object_size=self.object_size, num_targets=self.num_targets))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True if num_blocks > 0 else False, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=self.target_size + self.object_size,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, num_targets, reward_type, obs_type, viewpoint, target_size, object_size, robot, width, height)

        self.render_image_obs = False

    def _render_callback(self):

        self.sim.forward()

    def render(self, mode='human', size=None):
        self._render_callback()
        size = size if size is not None else (self.width, self.height)
        if mode == "seg":
            seg = self.sim.render(size[0], size[1], camera_name=self.viewpoint, segmentation=True)
            ids = seg[:, :, 1]

            # table to background
            id = self.sim.model.geom_name2id("table0")
            ids[ids == id] = -1

            for id in np.unique(ids):
                if id == -1:
                    continue

                name = self.sim.model.geom_id2name(id)
                if name is not None and ('forearm' in name or 'wrist' in name or 'elbow' in name or
                                         'upperarm' in name or 'gripper' in name or 'shoulder' in name):
                    ids[ids == id] = 254
                # robot to background
                elif name is not None and 'robot' in name:
                    ids[ids == id] = -1

            return (ids[::-1, :] + 1).astype(np.uint8)
        else:
            # original image is upside-down, so flip it
            return self.sim.render(size[0], size[1], camera_name=self.viewpoint)[::-1, :, :]

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.table_height_offset = self.sim.data.get_body_xpos('table0')[2]
        self.table_size = self.sim.model.geom_size[self.sim.model.geom_name2id('table0')]

    def _sample_goal(self):
        return np.array([0, 0, 0])

    def compute_reward(self, achieved_goal, goal, info):
        return 0

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def sample_pos(self, object_size):
        pos = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        pos[2] = self.table_height_offset + self.table_size[2] + object_size if pos[2] < self.table_height_offset + self.table_size[2] + object_size else pos[2]
        return pos.copy()

    def _reset_sim(self):
        assert self.num_blocks <= 17  # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize start position and colors of targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        ball_positions = []
        for i in range(self.num_targets):
            target_color = self.np_random.randint(0, 256, size=3)
            self.sim.model.site_rgba[self.sim.model.site_name2id(f'target{i}')][:3] = target_color / 255
            self.sim.model.site_rgba[self.sim.model.site_name2id(f'target{i}')][3] = 1

            pos_not_valid = True
            while pos_not_valid:
                target_xpos = self.sample_pos(self.target_size)

                pos_not_valid = False
                for position in ball_positions:
                    pos_not_valid = np.linalg.norm(target_xpos - position) < 2 * self.target_size
                    if pos_not_valid:
                        break

            ball_positions.append(target_xpos)

            site_id = self.sim.model.site_name2id(f'target{i}')
            self.sim.model.site_pos[site_id] = target_xpos - sites_offset[0]
        self.sim.forward()

        # Randomize start position and colors of objects.
        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_color = self.np_random.randint(0, 256, size=3)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(obj_name)][:3] = object_color / 255

            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                  size=2)
            while not np.all([np.linalg.norm(object_xypos - other_xpos) >= 2 * np.sqrt(2) * self.object_size for other_xpos in prev_obj_xpos]):
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