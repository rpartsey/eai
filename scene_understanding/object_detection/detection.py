import numpy as np
import habitat_sim
import quaternion

W = 1280
# H = 720
H = 1280
hfov = 90


sim_settings = {
    "scene_dataset": "data/replica_cad/replicaCAD.scene_dataset_config.json",
    "scene": "data/replica_cad/configs/scenes/apt_1.scene_instance.json",
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": W,
    "height": H,
    "enable_physics": True
}

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()

    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.hfov = hfov
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.hfov = hfov
    sensor_specs.append(depth_sensor_spec)

    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(sim_settings["default_agent"])

from votenet_habitat import votenet_build, votenet_detection, votenet_nms
votenet = votenet_build()

fx = W / np.tan(hfov / 2.) / 2
# fy = H / np.tan(hfov / 2.) / 2
fy = fx

K = np.array([
    [fx, 0., W / 2, 0.],
    [0., fy, H / 2, 0.],
    [0., 0., 1, 0],
    [0., 0., 0, 1]])

import glob
from PIL import Image

resultss = []
files = sorted(glob.glob("../../habitat-lab-display/saved_transformations/*.txt"), key=lambda x: tuple(int(y) for y in x.split("_")[-1][:-4].split("-")))


for idx, path in enumerate(files):
    with open(path) as f:
        nums = f.read().strip().split()
        T1 = np.array([float(x) for x in nums[:16]]).reshape((4, 4)).T
        T2 = np.array([float(x) for x in nums[16:]]).reshape((4, 4)).T

        T2[0:3, 0:3] = np.eye(3)

        translation = (T1 @ np.array([0, 0, 0, 1]))[:3]

        for j, beta in enumerate(np.linspace(0, 2 * np.pi, 1)):

            # T1[0:3, 0:3] = np.eye(3)
            # T2[0:3, 0:3] = quaternion.as_rotation_matrix(rotation)

            # rotation = quaternion.from_rotation_matrix((T2 @ T1)[:3, :3])
            rotation = quaternion.from_rotation_matrix(T1[:3, :3])
            # rotation = quaternion.from_rotation_matrix(T1[:3, :3][[0, 2, 1]][:, [0, 2, 1]])

            # Set agent state
            agent_state = habitat_sim.AgentState()
            agent_state.position = translation
            agent_state.rotation = rotation
            agent.set_state(agent_state)

            # Get agent state
            agent_state = agent.get_state()
            # print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            # print(agent_state.sensor_states)

            obs = sim.get_sensor_observations(0)

            if len(obs["depth_sensor"]) == 3:
                obs["depth_sensor"] = obs["depth_sensor"][...,0]

            observations = {
                "image_depth": obs["depth_sensor"],
                "image_depth_info": {
                    "width": W,
                    "height": H,
                    "matrix_intrinsics": K
                },
            }

            # Image.fromarray(obs["color_sensor"]).save(f"test{idx}-{j}.png")

            from matplotlib.cm import get_cmap
            dimg = (get_cmap("rainbow")((obs["depth_sensor"] / obs["depth_sensor"].max())) * 255).astype(np.uint8)
            Image.fromarray(dimg).save(f"test{idx}-{j}-depth.png")

            results = votenet_detection(votenet, observations, T1, T2)

            # print(translation, T1)
            print(f"Detected {len(results)} objects in the frame {idx}-{j}: {','.join(r['class'] for r in results)}")
            resultss.append(results)

DEFAULT_CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]

import json
res = votenet_nms(resultss, votenet, DEFAULT_CLASS_LIST)
with open("test.json", 'w') as f:
    json.dump(res, f, indent=4)
