from fetch_block_construction.envs.robotics.fetch.colors import get_colors
BASIC_COLORS = ["0 1 0", "1 1 0", "0.2 0.8 0.8", "0.8 0.2 0.8", "1.0 0.0 0.0", "0 0 0"]

base = '''
<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <compiler angle="radian" coordinate="local" meshdir="../stls/fetch" texturedir="../textures"></compiler>
    <option timestep="0.002">
        <flag warmstart="enable"></flag>
    </option>

    <include file="shared.xml"></include>

    <asset>
        {assets}
    </asset>

    <worldbody>
        <!--geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom-->
        <body name="floor0" pos="0.8 0.75 0">

        {target_sites}

        </body>

        <include file="{robot_file}.xml"></include>

        <-- Add additional cameras -->
        <body name="frontview_camera_body" pos="0 0 0">
				<camera euler="0. 1.0 1.571" fovy="60" name="frontview" pos="1.9 0.75 0.8"></camera>
		</body>
        <body name="topview_camera_body" pos="0 0 0">
				<camera euler="0. 0 0." fovy="60" name="topview" pos="1.3 0.75 1.25"></camera>
		</body>

        <body pos="1.3 0.75 0.2" name="table0">
            <geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat" name="table0"></geom>
        </body>
        
        {object_bodies}
        
        <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
    </actuator>
</mujoco>
'''

def generate_multi_camera_xml(num_blocks: int, robot: str, task='vbb', target_size=0.02, object_size=0.025, num_targets=0):
    if num_blocks <= (6 if not task == 'reach' else 5):
        colors = BASIC_COLORS[:max(num_blocks, num_targets)]
    else:
        colors = get_colors(max(num_blocks, num_targets))

    site_base = '<site name="target{id}" pos="0 0 0.5" size="{target_size} {target_size} {target_size}" rgba="{color} 0.3" type="sphere"></site>' \
        if not task == 'reach' else '<site name="distractor{id}" pos="0 0 0.5" size="{target_size} {target_size} {target_size}" rgba="{color} 1" type="sphere"></site>'
    block_base = '''<body name="object{id}" pos="0.025 0.025 0.025">
        <joint name="object{id}:joint" type="free" damping="0.01"></joint>
        <geom size="{object_size} {object_size} {object_size}" type="box" condim="3" name="object{id}" material="block{id}_mat" mass="2"></geom>
        <site name="object{id}" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
    </body>'''
    asset_base = '<material name="block{id}_mat" specular="0" shininess="0.5" reflectance="0" rgba="{color} 1"></material>'

    robot_file = "simplified_robot" if robot == "simplified" else "robot"

    sites = []
    block_bodies = []
    assets = []

    if task == 'reach':
        sites.append(f'<site name="target0" pos="0 0 0.5" size="{target_size} {target_size} {target_size}" rgba="1 0 0 1" type="sphere"></site>')
        if "1.0 0.0 0.0" in colors:
            colors.remove("1.0 0.0 0.0")
        if "1 0 0" in colors:
            colors.remove("1 0 0")

    if task == 'pick_place':
        sites.append(f'<site name="target0" pos="0 0 0.5" size="{target_size} {target_size} {target_size}" rgba="1 0 0 1" type="sphere"></site>')

    for i in range(num_blocks):
        if task == 'reach':
            sites.append(site_base.format(**dict(id=i, target_size=target_size, color=colors[i])))
        else:
            if task == 'vbb':
                sites.append(site_base.format(**dict(id=i, target_size=target_size, color=colors[i])))
            block_bodies.append(block_base.format(**dict(id=i, object_size=object_size)))
            assets.append(asset_base.format(**dict(id=i, color=colors[i])))

    if task == 'render':
        for i in range(num_targets):
            sites.append(site_base.format(**dict(id=i, target_size=target_size, color=colors[i])))

    return base.format(**dict(assets="\n".join(assets), target_sites="\n".join(sites), robot_file=robot_file, object_bodies="\n".join(block_bodies))) \
        if not task == 'reach' else base.format(**dict(assets="", target_sites="\n".join(sites), robot_file=robot_file, object_bodies=""))
