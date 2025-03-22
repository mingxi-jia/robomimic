"""
Adapted from robomimic/scripts/dataset_states_to_obs.py
"""

import argparse
import json
import multiprocessing
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from multiprocessing import Array

import h5py
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robosuite
from robomimic.envs.env_base import EnvBase
from tqdm import tqdm
import cv2
import re

from robomimic.config.dataset_config import Objective, load_config, parse_config_overrides
from robomimic.utils.performance_utils import check_memory_and_time
from robomimic.utils.python_utils import numpy_array_to_string

multiprocessing.set_start_method("spawn", force=True)

ROBOSUITE_XML_DIR_RELATIVE_PATH="models/assets/arenas/"

def dict_obs_to_JPEG(dict_obs: dict):
    for k, v in dict_obs.items():
        is_image = (len(v.shape) == 3)
        if is_image:
            _, JPEG = cv2.imencode('.jpeg', v)
            dict_obs[k] = JPEG
    return dict_obs

def add_cameras_to_state(xml_string: str, config) -> str:
    root = ET.fromstring(xml_string)
    
    worldbody = root.find('.//worldbody')
    
    cameras = worldbody.findall('./camera')
    
    for camera_name, camera_config in config.cameras.items():
        
        new_camera = ET.Element('camera')
        new_camera.set('mode', 'fixed')
        new_camera.set('name', camera_name)
        new_camera.set('pos', numpy_array_to_string(camera_config['pos']))
        new_camera.set('quat', numpy_array_to_string(camera_config['quat_wxyz']))
        
        last_camera_idx = list(worldbody).index(cameras[-1])
        worldbody.insert(last_camera_idx + 1, new_camera)
    
    return ET.tostring(root, encoding='unicode', method='xml')

def extract_trajectory(
        config: SimpleNamespace, 
        env_meta: dict, 
        initial_state: dict,
        states: np.array, 
        actions: np.array
    ):
    done_mode = config.done_mode

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(config.cameras.keys()),
        camera_height=config.camera_resolution,
        camera_width=config.camera_resolution,
        reward_shaping=False,
        use_depth_obs=True
    )

    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    env.reset()

    initial_state["model"] = add_cameras_to_state(initial_state["model"], config)

    traj_len = states.shape[0]

    traj = dict(
        obs=[None] * (traj_len),
        rewards=[None] * (traj_len),
        dones=[None] * (traj_len),
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )

    # Iterate over the time steps
    for t in tqdm(range(0, traj_len)):
        
        if t == 0:
            obs = env.reset_to(initial_state)
        else:
            # reset to simulator state to get observation
            obs = env.reset_to({"states": states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # Done: 0: every successful timestep, 1: at the end of the traj, 2: both
        if (done_mode == 1) or (done_mode == 2):
            done = (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            done = env.is_success()["task"]
        
        done = int(done)

        # del obs['voxels']
        traj["obs"][t] = obs
        traj["rewards"][t] = r
        traj["dones"][t] = done

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    if config.camera_obs_modality == "rgbd": # NOTE: this changes the dtype of np array to float32. Increases dataset size.
        for camera in config.cameras:
            traj["obs"][f"{camera}_rgbd"] = np.concatenate(
                [
                    traj["obs"][f"{camera}_image"],
                    traj["obs"][f"{camera}_depth"],
                ],
                axis=-1,
            )
            del traj["obs"][f"{camera}_image"]
            del traj["obs"][f"{camera}_depth"]
    elif config.camera_obs_modality == "rgb":
        for camera in config.cameras:
            del traj["obs"][f"{camera}_depth"]
            del traj["obs"][f"{camera}_rgbd"]
    elif config.camera_obs_modality == "separate":
        pass
    else:
        raise ValueError("Invalid camera obs modality")

    return traj


def worker(input):
    args, env_meta, initial_state, states, actions = input

    traj = extract_trajectory(
        args,
        env_meta=env_meta,
        initial_state=initial_state,
        states=states,
        actions=actions
    )

    return traj

def store_obs_as_nested_folders(folder_path, obs_dict: dict, compress=False):
    def save_compressed_image(image, path):
        with open(path, 'wb') as f:
            _, JPEG = cv2.imencode('.jpeg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            f.write(JPEG.tobytes())

    for k, v in obs_dict.items():
        traj_length = len(v)
        if re.fullmatch(r"^\w+_image$", k):
            image_folder_path = os.path.join(folder_path, k)
            os.makedirs(image_folder_path, exist_ok=True)
            for i in range(traj_length):
                if compress:
                    save_compressed_image(v[i], os.path.join(image_folder_path, f"{i}.jpg"))
                else:
                    cv2.imwrite(os.path.join(image_folder_path, f"{i}.png"), cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))
        elif re.fullmatch(r"^\w+_depth$", k):
            image_folder_path = os.path.join(folder_path, k)
            os.makedirs(image_folder_path, exist_ok=True)
            for i in range(traj_length):
                if compress:
                    save_compressed_image(v[i], os.path.join(image_folder_path, f"{i}.jpg"))
                else:
                    np.save(os.path.join(image_folder_path, f"{i}.png"), v[i])
                    depth_min, depth_max = ObsUtils.DEPTH_MINMAX[k]
                    depth_vis = (v[i] - depth_min) / (depth_max - depth_min)
                    depth_vis = np.clip(depth_vis, 0., 1.) * 255
                    cv2.imwrite(os.path.join(image_folder_path, f"{i}.png"), depth_vis.astype(np.uint8))
        elif k=='pcd':
            continue
        else:
            np.save(os.path.join(folder_path, f"{k}.npy"), v)

def store_data_as_npys(path, data):
    np.save(path+'.npy', data)

def generate_obs_for_dataset_parallel(config: SimpleNamespace) -> None:
    env_meta = FileUtils.get_env_metadata_from_dataset(config.dataset_path)

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(config.dataset_path, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    output_dir = config.output_dir
    output_folder = config.output_folder
    output_path = os.path.join(output_dir, output_folder)

    if os.path.exists(output_path):
        demo_folder_list = os.listdir(output_path)
        pattern = r"^demo_\d+$"
        # Filter out strings that match the pattern
        filtered_strings = [s for s in demo_folder_list if re.fullmatch(pattern, s)]
        start_ind = len(filtered_strings)
    else:
        os.makedirs(output_path, exist_ok=True)
        start_ind = 0

    demos = demos[: start_ind + config.num_demos]
    num_workers = config.num_workers
    total_samples = 0

    for i in tqdm(range(start_ind, len(demos), num_workers)):
        current_demos = demos[i : min(i + num_workers, len(demos))]
        initial_state_list = []
        states_list = []
        actions_list = []

        for ep in current_demos:
            states_list.append(f["data/{}/states".format(ep)][()])
            actions_list.append(f["data/{}/actions".format(ep)][()])

            initial_state = dict(states=states_list[-1][0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

            initial_state_list.append(initial_state)

        inputs = [
            (config, env_meta, initial_state_list[j], states_list[j], actions_list[j])
            for j in range(len(current_demos))
        ]

        with multiprocessing.Pool(num_workers) as p:
            trajs = p.map(worker, inputs)

        del inputs
        del initial_state_list
        del states_list
        del actions_list

        for j in range(len(current_demos)):
            ep = current_demos[j]
            traj = trajs[j]

            demo_path = os.path.join(output_path, ep)
            os.makedirs(demo_path, exist_ok=True)
            store_obs_as_nested_folders(os.path.join(demo_path, "obs"), traj['obs'])
            store_data_as_npys(os.path.join(demo_path, "actions"), np.array(traj["actions"]))
            store_data_as_npys(os.path.join(demo_path, "states"), np.array(traj["states"]))
            store_data_as_npys(os.path.join(demo_path, "rewards"), np.array(traj["rewards"]))
            store_data_as_npys(os.path.join(demo_path, "dones"), np.array(traj["dones"]))

            # episode metadata
            if is_robosuite_env:
                store_data_as_npys(os.path.join(demo_path, "model_file"), traj["initial_state_dict"]["model"])

            store_data_as_npys(os.path.join(demo_path, "num_samples"), np.array(traj["actions"].shape[0]))
            total_samples += traj["actions"].shape[0]
            print(
                "{}: wrote {} transitions to path {}".format(
                    ep, traj["actions"].shape[0], output_path
                )
            )

        del trajs

        print(f"done {len(demos)} demos")

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(config.cameras.keys()),
        camera_height=config.camera_resolution,
        camera_width=config.camera_resolution,
        reward_shaping=False,
        use_depth_obs=True
    )
    # global metadata
    store_data_as_npys(os.path.join(output_path, "env_args"), np.array(json.dumps(env.serialize(), indent=4)))
    store_data_as_npys(os.path.join(output_path, "total"), np.array(total_samples))
    store_data_as_npys(os.path.join(output_path, "camera_extrinsics"), np.array(config.cameras))
    store_data_as_npys(os.path.join(demo_path, "num_samples"), np.array(traj["actions"].shape[0]))
    print("Wrote {} demos to {}".format(len(demos), output_path))

    f.close()


@check_memory_and_time
def generate_multicamera_obs_dataset(config: SimpleNamespace) -> None:
    print("Setting up cameras...")

    print("Generating observations...")
    try:
        generate_obs_for_dataset_parallel(config)
    except Exception as e:
        print(e)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="lift",
        help="Name of config file."
    )

    parser.add_argument(
        "--debug", 
        action="store_true",
        default=False,
        help="Debug mode. Only 1 demo if true."
    )

    parser.add_argument(
        "--overrides", 
        type=str, 
        nargs='*', 
        default=[], 
        help="List of configuration overrides."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    overrides = parse_config_overrides(args.overrides)

    config = load_config(
        Objective.DATASET_GENERATION, 
        args.config,
        overrides
    )
    
    if config.camera_obs_modality not in ["rgbd", "separate", "rgb"]:
        raise ValueError("Invalid camera obs modality")
    
    if args.debug:
        config.num_demos = 1
        config.num_workers = 1
    
    config.env_xml_path = os.path.join(
        robosuite.__path__[0], 
        ROBOSUITE_XML_DIR_RELATIVE_PATH, 
        config.env_xml_filename
    )
    print(f"Using env xml path: {config.env_xml_path}")
    
    config.dataset_path = os.path.join(
        config.dataset_dir, 
        config.dataset_file
    )

    output_file = os.path.join(
        "/".join(config.dataset_file.split("/")[:-1]), 
        ("test_" if args.debug else "") 
        + f"multicamera_{config.camera_obs_modality}_"
        + config.dataset_file.split("/").pop()
    )
    config.output_path = os.path.join(config.output_dir, output_file)

    # Robosuite preset cameras
    config.cameras = {
        "spaceview": {
            "pos": np.array([0.85, 0, 1.55]),
            "quat_wxyz": np.array([0.6341848, 0.3127453, 0.3127453, 0.6341848]),
        },
    }

    generate_multicamera_obs_dataset(config)
