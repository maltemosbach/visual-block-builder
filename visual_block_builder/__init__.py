import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)

for num_blocks in range(1, 25):
    for reward_type in ['sparse', 'dense', 'incremental', 'block1only']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for stack_only in [True, False]:
                for case in ["Singletower", "Pyramid", "Multitower", "All"]:
                    for viewpoint in ["frontview", "topview", "external_camera_0"]:
                        for robot in ["default", "simplified"]:
                            initial_qpos = {
                                'robot0:slide0': 0.405,
                                'robot0:slide1': 0.48,
                                'robot0:slide2': 0.0,
                                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                            }

                            for i in range(num_blocks):
                                initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]
                            kwargs = {
                                'reward_type': reward_type,
                                'initial_qpos': initial_qpos,
                                'num_blocks': num_blocks,
                                'obs_type': obs_type,
                                'stack_only': stack_only,
                                'case': case,
                                'viewpoint': viewpoint,
                                'robot': robot,
                            }

                            register(
                                id='VisualBlockBuilder_{}Blocks_{}Reward_{}Obs_{}Stackonly_{}Case_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, reward_type, obs_type, stack_only, case, viewpoint, robot]]),
                                entry_point='visual_block_builder.env:VisualBlockBuilderEnv',
                                kwargs=kwargs,
                                max_episode_steps=50 * num_blocks,
                            )


for num_distractors in range(0, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for size in ["small", "medium", "large"]:
                        initial_qpos = {
                            'robot0:slide0': 0.405,
                            'robot0:slide1': 0.48,
                            'robot0:slide2': 0.0,
                        }

                        kwargs = {
                            'reward_type': reward_type,
                            'initial_qpos': initial_qpos,
                            'num_distractors': num_distractors,
                            'obs_type': obs_type,
                            'case': 'Specific',
                            'viewpoint': viewpoint,
                            'robot': robot,
                            'target_size': size,
                        }

                        register(
                            id='ReachSpecificTarget_{}Distractors_{}Targets_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_distractors, size, reward_type, obs_type, viewpoint, robot]]),
                            entry_point='visual_block_builder.env:ReachTargetEnv',
                            kwargs=kwargs,
                            max_episode_steps=50,
                        )

for num_distractors in range(2, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for size in ["small", "medium", "large"]:
                        initial_qpos = {
                            'robot0:slide0': 0.405,
                            'robot0:slide1': 0.48,
                            'robot0:slide2': 0.0,
                        }

                        kwargs = {
                            'reward_type': reward_type,
                            'initial_qpos': initial_qpos,
                            'num_distractors': num_distractors,
                            'obs_type': obs_type,
                            'case': 'Distinct',
                            'viewpoint': viewpoint,
                            'robot': robot,
                            'target_size': size,
                        }

                        register(
                            id='ReachDistinctTarget_{}Distractors_{}Targets_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_distractors, size, reward_type, obs_type, viewpoint, robot]]),
                            entry_point='visual_block_builder.env:ReachTargetEnv',
                            kwargs=kwargs,
                            max_episode_steps=50,
                        )

for num_distractors in range(0, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for size in ["small", "medium", "large"]:
                        initial_qpos = {
                            'robot0:slide0': 0.405,
                            'robot0:slide1': 0.48,
                            'robot0:slide2': 0.0,
                        }

                        kwargs = {
                            'reward_type': reward_type,
                            'initial_qpos': initial_qpos,
                            'num_distractors': num_distractors,
                            'obs_type': obs_type,
                            'case': 'Random',
                            'viewpoint': viewpoint,
                            'robot': robot,
                            'target_size': size,
                        }

                        register(
                            id='ReachRandomTarget_{}Distractors_{}Targets_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_distractors, size, reward_type, obs_type, viewpoint, robot]]),
                            entry_point='visual_block_builder.env:ReachTargetEnv',
                            kwargs=kwargs,
                            max_episode_steps=200,
                        )

for num_blocks in range(1, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for target_size in ["small", "medium", "large"]:
                        for object_size in ["small", "medium", "large"]:
                            initial_qpos = {
                                'robot0:slide0': 0.405,
                                'robot0:slide1': 0.48,
                                'robot0:slide2': 0.0,
                                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                            }

                            for i in range(num_blocks):
                                initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

                            kwargs = {
                                'reward_type': reward_type,
                                'initial_qpos': initial_qpos,
                                'num_blocks': num_blocks,
                                'obs_type': obs_type,
                                'case': "Specific",
                                'viewpoint': viewpoint,
                                'robot': robot,
                                'target_size': target_size,
                                'object_size': object_size,
                            }

                            register(
                                id='PickAndPlaceSpecificBlock_{}Blocks_{}Target_{}Objects_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, target_size, object_size, reward_type, obs_type, viewpoint, robot]]),
                                entry_point='visual_block_builder.env:PickAndPlaceBlockEnv',
                                kwargs=kwargs,
                                max_episode_steps=100,
                            )

for num_blocks in range(3, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for target_size in ["small", "medium", "large"]:
                        for object_size in ["small", "medium", "large"]:
                            initial_qpos = {
                                'robot0:slide0': 0.405,
                                'robot0:slide1': 0.48,
                                'robot0:slide2': 0.0,
                                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                            }

                            for i in range(num_blocks):
                                initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i * .06, 1., 0., 0., 0.]

                            kwargs = {
                                'reward_type': reward_type,
                                'initial_qpos': initial_qpos,
                                'num_blocks': num_blocks,
                                'obs_type': obs_type,
                                'case': "Distinct",
                                'viewpoint': viewpoint,
                                'robot': robot,
                                'target_size': target_size,
                                'object_size': object_size,
                            }

                            register(
                                id='PickAndPlaceDistinctBlock_{}Blocks_{}Target_{}Objects_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, target_size, object_size, reward_type, obs_type, viewpoint, robot]]),
                                entry_point='visual_block_builder.env:PickAndPlaceBlockEnv',
                                kwargs=kwargs,
                                max_episode_steps=100,
                            )

for num_blocks in range(1, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for target_size in ["small", "medium", "large"]:
                        for object_size in ["small", "medium", "large"]:
                            initial_qpos = {
                                'robot0:slide0': 0.405,
                                'robot0:slide1': 0.48,
                                'robot0:slide2': 0.0,
                                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                            }

                            for i in range(num_blocks):
                                initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

                            kwargs = {
                                'reward_type': reward_type,
                                'initial_qpos': initial_qpos,
                                'num_blocks': num_blocks,
                                'obs_type': obs_type,
                                'case': "Random",
                                'viewpoint': viewpoint,
                                'robot': robot,
                                'target_size': target_size,
                                'object_size': object_size,
                            }

                            register(
                                id='PickAndPlaceRandomBlock_{}Blocks_{}Target_{}Objects_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, target_size, object_size, reward_type, obs_type, viewpoint, robot]]),
                                entry_point='visual_block_builder.env:PickAndPlaceBlockEnv',
                                kwargs=kwargs,
                                max_episode_steps=200,
                            )


for num_blocks in range(1, 18):
    for reward_type in ['sparse', 'dense']:
        for obs_type in ['dictimage', 'np', 'dictstate']:
            for viewpoint in ["frontview", "topview", "external_camera_0"]:
                for robot in ["default", "simplified"]:
                    for target_size in ["small", "medium", "large"]:
                        for object_size in ["small", "medium", "large"]:
                            initial_qpos = {
                                'robot0:slide0': 0.405,
                                'robot0:slide1': 0.48,
                                'robot0:slide2': 0.0,
                                'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
                            }

                            for i in range(num_blocks):
                                initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

                            kwargs = {
                                'reward_type': reward_type,
                                'initial_qpos': initial_qpos,
                                'num_blocks': num_blocks,
                                'obs_type': obs_type,
                                'viewpoint': viewpoint,
                                'robot': robot,
                                'target_size': target_size,
                                'object_size': object_size,
                            }

                            register(
                                id='PickAndPlaceSort_{}Blocks_{}Target_{}Objects_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, target_size, object_size, reward_type, obs_type, viewpoint, robot]]),
                                entry_point='visual_block_builder.env:PickAndPlaceSortEnv',
                                kwargs=kwargs,
                                max_episode_steps=200,
                            )

for num_blocks in range(0, 18):
    for num_targets in range(0, 18):
        for reward_type in ['sparse', 'dense']:
            for obs_type in ['dictimage', 'np', 'dictstate']:
                for viewpoint in ["frontview", "topview", "external_camera_0"]:
                    for robot in ["default", "simplified"]:
                        for target_size in ["small", "medium", "large"]:
                            for object_size in ["small", "medium", "large"]:
                                initial_qpos = {
                                    'robot0:slide0': 0.405,
                                    'robot0:slide1': 0.48,
                                    'robot0:slide2': 0.0,
                                }

                                for i in range(num_blocks):
                                    initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

                                kwargs = {
                                    'reward_type': reward_type,
                                    'initial_qpos': initial_qpos,
                                    'num_blocks': num_blocks,
                                    'obs_type': obs_type,
                                    'viewpoint': viewpoint,
                                    'robot': robot,
                                    'target_size': target_size,
                                    'object_size': object_size,
                                    'num_targets': num_targets,
                                }

                                register(
                                    id='Render_{}Blocks_{}_{}Targets_{}Objects_{}Reward_{}Obs_{}Viewpoint{}Robot-v1'.format(*[kwarg.title() if isinstance(kwarg, str) else kwarg for kwarg in [num_blocks, num_targets, target_size, object_size, reward_type, obs_type, viewpoint, robot]]),
                                    entry_point='visual_block_builder.env:RenderEnv',
                                    kwargs=kwargs,
                                    max_episode_steps=200,
                                )
