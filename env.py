# env.py
# A pure-execution palletization environment.
# - The heuristic is responsible for producing the FINAL action (logits vector).
# - The environment does NOT compute feasibility mask, does NOT adjust (x, y).
# - The environment decodes action -> places box -> sim forward -> stability check -> state update.

import copy
import random
import argparse
import numpy as np
import imageio
import torch
import os
from datetime import datetime

from scipy.special import softmax

import gymnasium as gym
from gymnasium import spaces

from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.transform_utils import convert_quat
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

from helpers.task_config import TaskConfig

SIM_TIMESTEP = 0.002


def _resolve_video_path(path: str):

    if path is None:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    base, ext = os.path.splitext(path)
    if ext == "":
        ext = ".mp4"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}__{timestamp}{ext}"


class BoxPlanning(SingleArmEnv):

    n_frame = 0

    def __init__(
        self,
        save_video_path=None,
        device=torch.device("cuda:0"),
        init_box_pose_path=None,
        control_freq=20,
        horizon=100,
        ignore_done=True,
    ):
        # === Table config ===
        self.table_full_size = TaskConfig.table.full_size
        self.table_friction = TaskConfig.table.friction
        self.table_offset = np.array(TaskConfig.table.offset)

        # === Pallet config ===
        self.pallet_size = TaskConfig.pallet.size
        self.pallet_position = self.table_offset + TaskConfig.pallet.relative_table_displacement

        # === Task config ===
        self.N_visible_boxes = TaskConfig.buffer_size
        self.max_pallet_height = TaskConfig.pallet.max_pallet_height
        self.bin_size = TaskConfig.bin_size
        self.pallet_size_discrete = (np.array(self.pallet_size)[:2] / self.bin_size).astype(int)
        self.n_properties = TaskConfig.box.n_properties
        self.n_box_types = TaskConfig.box.n_type

        self.stable_thres = 0.02
        self.random_generator = None

        self.device = device
        self.init_box_pose_path = init_box_pose_path

        # === Controller config ===
        controller_configs = load_controller_config(custom_fpath="./helpers/controller.json")

        # === Video config ===
        self.save_video = save_video_path is not None
        self.writer = None
        if self.save_video:
            final_video_path = _resolve_video_path(save_video_path)
            self.writer = imageio.get_writer(final_video_path, fps=control_freq)
            print(f"[Video] Saving to: {final_video_path}")

        super().__init__(
            robots=["Panda"],
            controller_configs=controller_configs,
            initialization_noise=None,
            horizon=horizon,
            has_renderer=False,
            has_offscreen_renderer=self.save_video,
            use_camera_obs=self.save_video,
            control_freq=control_freq,
            ignore_done=ignore_done,
        )
    # ---------------------------
    # Model construction
    # ---------------------------

    def creat_box(self, box_type, box_name):

        physics_mode = getattr(TaskConfig, "physics", None)
        physics_mode = physics_mode.mode if physics_mode is not None else "soft"

        if physics_mode == "rigid":
            # ===============================
            # RIGID MODE (geometry / heuristic stage)
            # ===============================
            density = 10.0
            solref = [0.001, 1.0]
            solimp = [0.99, 0.99, 0.001]
            friction = (1.0, 0.005, 0.0001)

        elif physics_mode == "soft":
            # ===============================
            # SOFT MODE (physics-aware stage)
            # ===============================
            cfg = TaskConfig.box.type_dict[box_type]
            density = cfg["density"]
            solref = [0.02, cfg["softness"]]
            solimp = [0.9, 0.95, 0.001]
            friction = cfg["friction"]

        else:
            raise ValueError(f"Unknown physics mode: {physics_mode}")

        box = BoxObject(
            name=box_name,
            size=TaskConfig.box.type_dict[box_type]["size"],
            material=TaskConfig.box.type_dict[box_type]["material"],
            friction=friction,
            density=density,
            solref=solref,
            solimp=solimp,
        )
        return box

    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Load boxes
        if self.init_box_pose_path is not None:
            self.load_box_record_pose()
        else:
            self.load_box_random_pose()

        # Load pallet
        self.pallet = BoxObject(name="pallet", size=np.array(self.pallet_size) / 2, material=TaskConfig.pallet.material)
        self.pallet.get_obj().set("pos", array_to_string(self.pallet_position))

        # Put together
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pallet] + [item for sublist in self.box_obj_list for item in sublist],
        )

    def load_box_record_pose(self):
        self.box_init_pose = np.load(self.init_box_pose_path).tolist()
        self.total_box_number = 0
        self.box_obj_list = [[] for _ in range(self.n_box_types)]

        for i in range(self.n_box_types):
            box_type = i + 1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.creat_box(box_type, box_name)

                init_pose = np.array(self.box_init_pose[self.total_box_number])
                box_obj.get_obj().set("pos", array_to_string(init_pose[:3]))
                box_obj.get_obj().set("quat", array_to_string(T.convert_quat(init_pose[3:], to="wxyz")))

                self.box_obj_list[i].append(box_obj)
                self.total_box_number += 1

    def load_box_random_pose(self):
        self.total_box_number = 0
        self.box_obj_list = [[] for _ in range(self.n_box_types)]

        for i in range(self.n_box_types):
            box_type = i + 1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.creat_box(box_type, box_name)

                init_x = random.uniform(-0.3, 0.3)
                init_y = random.uniform(-0.8, -0.3)
                init_z = random.uniform(0, 0.05) + (self.n_box_types - 1 - i) * 0.05
                box_obj.get_obj().set(
                    "pos",
                    array_to_string(self.table_offset - box_obj.bottom_offset + np.array([init_x, init_y, init_z])),
                )

                self.box_obj_list[i].append(box_obj)
                self.total_box_number += 1

    def _setup_references(self):
        super()._setup_references()

        self.boxes_body_id = [[] for _ in range(self.n_box_types)]
        self.boxes_id_to_index = {}
        self.boxes_names = []
        self.boxes_ids = []
        self.id_to_box_obj = {}
        self.id_to_properties = {}

        for i in range(self.n_box_types):
            for j in range(len(self.box_obj_list[i])):
                box_id = self.sim.model.body_name2id(self.box_obj_list[i][j].root_body)
                self.boxes_names.append(self.box_obj_list[i][j].root_body[:-5])
                self.boxes_body_id[i].append(box_id)
                self.boxes_id_to_index[box_id] = [i, j]
                self.boxes_ids.append(box_id)
                self.id_to_box_obj[box_id] = self.box_obj_list[i][j]
                box_property = list(np.array(self.box_obj_list[i][j].size) * 100) + [self.box_obj_list[i][j].density / 1000]
                self.id_to_properties[box_id] = np.array(box_property, dtype=np.float32)

    # ---------------------------
    # Action decoding
    # ---------------------------

    def choose_index(self, action):
        """Pick which buffer index to use from logits."""
        sample_logits = action[: self.N_visible_boxes]
        pick_likelihood = softmax(sample_logits)
        sample_index = int(np.argmax(pick_likelihood))
        return sample_index

    def get_orientation(self, action):
        """Pick one of 6 orientation IDs from logits; currently forced to 0"""
        sample_ori_logits = action[self.N_visible_boxes : self.N_visible_boxes + 6]
        sample_ori = int(np.argmax(sample_ori_logits))

        # Your original code forces orientation to 0; keep it for compatibility
        sample_ori = 0

        orders = {
            0: np.array([0, 1, 2]),
            1: np.array([0, 2, 1]),
            2: np.array([1, 0, 2]),
            3: np.array([1, 2, 0]),
            4: np.array([2, 0, 1]),
            5: np.array([2, 1, 0]),
        }
        target_quat = self.compute_target_quat_from_order(sample_ori)
        return orders[sample_ori], target_quat

    def compute_target_quat_from_order(self, order):
        if order == 0:
            rotm = np.eye(3)
        elif order == 1:
            rotm = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        elif order == 2:
            rotm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        elif order == 3:
            rotm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        elif order == 4:
            rotm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        elif order == 5:
            rotm = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

        return T.mat2quat(rotm)  # (xyzw)

    def get_target_position(self, action, box_size_after_rotate):
        """
        IMPORTANT: No feasible_points projection here.
        The heuristic is responsible for choosing final (x,y).
        Env only computes z based on current pallet occupancy and maps to world coordinates.
        """
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])

        # x
        x_logits = action[self.N_visible_boxes + 6 : self.N_visible_boxes + 6 + X]
        x = int(x_logits.argmax())

        # y
        y_logits = action[self.N_visible_boxes + 6 + X : self.N_visible_boxes + 6 + X + Y]
        y = int(y_logits.argmax())

        # clamp (safety)
        x = int(np.clip(x, 0, X - 1))
        y = int(np.clip(y, 0, Y - 1))

        # Determine z position based on pallet situation
        x2 = int(min(x + box_size_after_rotate[0], X))
        y2 = int(min(y + box_size_after_rotate[1], Y))

        place_area = self.obs["pallet_obs_density"][x:x2, y:y2, :]
        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

        # World coordinates (box center)
        target_x = self.pallet_position[0] - self.pallet_size[0] / 2 + x * self.bin_size + box_size_after_rotate[0] * self.bin_size / 2
        target_y = self.pallet_position[1] - self.pallet_size[1] / 2 + y * self.bin_size + box_size_after_rotate[1] * self.bin_size / 2
        target_z = self.pallet_position[2] + self.pallet_size[2] / 2 + z * self.bin_size + box_size_after_rotate[2] * self.bin_size / 2
        target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)

        return target_pos, (x, y, z)

    # ---------------------------
    # Execution & state update
    # ---------------------------

    def place_box(self, box_obj, target_pos, target_quat):
        """Teleport box to target pose. quat: xyzw"""
        self.sim.data.set_joint_qpos(
            box_obj.joints[0],
            np.concatenate([target_pos, T.convert_quat(target_quat, to="wxyz")]),
        )
        self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

    def sim_forward(self, steps):
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def check_stable(self):
        for box_id in self.boxes_on_pallet_id:
            box_cur_position = self.get_box_pose(box_id)[:3]
            box_target_position = self.boxes_on_pallet_target_pose[box_id][:3]
            if np.linalg.norm(box_cur_position - box_target_position) > self.stable_thres:
                return False
        return True

    def get_box_pose(self, box_id):
        box_pos = np.array(self.sim.data.body_xpos[box_id])
        box_quat = convert_quat(np.array(self.sim.data.body_xquat[box_id]), to="xyzw")
        return np.hstack((box_pos, box_quat))

    def update_obs_buffer(self):
        boxes_in_buffer = np.zeros(self.N_visible_boxes * self.n_properties, dtype=np.float32)
        for i in range(min(len(self.unplaced_box_ids), self.N_visible_boxes)):
            box_property = self.id_to_properties[self.unplaced_box_ids[i]]
            boxes_in_buffer[self.n_properties * i : self.n_properties * (i + 1)] = box_property
        self.obs["buffer"] = boxes_in_buffer
        return boxes_in_buffer

    def save_frame(self):
        self._update_observables()
        video_img = self.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        self.writer.append_data(video_img)

    def record_pallet(self, sample_boxid, target_quat, size_after_rotate):
        record_data = {
            "pallet_config": self.boxes_on_pallet_target_pose,
            "pallet_obs_density": self.obs["pallet_obs_density"].copy(),
            "to_place_id": sample_boxid,
            "to_place_quat": target_quat.copy(),  # xyzw
            "size_after_rotate": size_after_rotate.copy(),
            "to_place_density": float(self.id_to_properties[sample_boxid][3]),
        }
        return record_data

    # ---------------------------
    # Reward (optional)
    # ---------------------------

    def reward_box_size(self, box_size_discrete):
        box_vol = float(np.prod(box_size_discrete))
        denom = float(self.pallet_size_discrete[0] * self.pallet_size_discrete[1] * self.max_pallet_height)
        return box_vol / denom

    def reward_func(self, termination_reason, box_size_discrete=None, position_discrete=None):
        """
        termination_reason:
            0 -> ongoing
            2 -> unstable (failed placement)
            3 -> success
            4 -> invalid action (picked empty buffer slot)
        """
        if termination_reason == 0 or termination_reason == 3:
            r1 = self.reward_box_size(box_size_discrete)
        else:
            r1 = 0.0

        info = {"reward_box_size": r1, "termination_reason": int(termination_reason)}
        return float(r1), info

    # ---------------------------
    # Gym-like API
    # ---------------------------

    def reset(self):
        _ = super().reset()

        self.unplaced_box_ids = copy.copy(self.boxes_ids)

        self.obs = {}
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])
        self.obs["pallet_obs_density"] = np.zeros((X, Y, self.max_pallet_height), dtype=np.float32)
        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}

        return self.obs

    def reinit(self, random_generator: np.random.Generator):
        """
        Reinitialize with deterministic randomness support (for wrapper reset).
        """
        self.init_box_pose()
        self.unplaced_box_ids = copy.copy(self.boxes_ids)

        if self.random_generator is None:
            self.random_generator = random_generator

        # shuffle order
        self.random_generator.shuffle(self.unplaced_box_ids)

        self.obs = {}
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])
        self.obs["pallet_obs_density"] = np.zeros((X, Y, self.max_pallet_height), dtype=np.float32)
        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}

        return self.obs

    def init_box_pose(self):
        """
        If init_box_pose_path provided, the model already has poses at load time.
        Here we just place them again (useful for reinit).
        """
        if not hasattr(self, "box_init_pose"):
            return

        for i in range(self.total_box_number):
            box_id = self.boxes_ids[i]
            init_pose = np.array(self.box_init_pose[i])
            box_obj = self.id_to_box_obj[box_id]
            self.place_box(box_obj, init_pose[:3], init_pose[3:])

    def step(self, action):
        """
        The heuristic provides the final action logits.
        Env decodes them and executes exactly that decision.
        """
        # 1) choose which buffer index
        sample_box_index = self.choose_index(action)

        # invalid: buffer slot does not exist
        if sample_box_index >= len(self.unplaced_box_ids):
            # You can decide whether to terminate or not. Here we keep episode running.
            reward, info = self.reward_func(termination_reason=4)
            done = False
            if self.save_video:
                self.save_frame()
            info["record_data"] = None
            return self.obs, reward, done, info

        # 2) get selected box data
        sample_box_id = self.unplaced_box_ids[sample_box_index]
        sample_box = self.id_to_box_obj[sample_box_id]

        sample_box_size = (np.array(sample_box.size) * 2 / self.bin_size).astype(int)  # discretized size
        sample_box_density = self.id_to_properties[sample_box_id][3]

        # 3) orientation (currently forced to 0) & rotated size
        orientation, target_quat = self.get_orientation(action)
        box_size_after_rotate = sample_box_size[orientation]

        # 4) record pallet state (optional)
        record_data = self.record_pallet(sample_box_id, target_quat, box_size_after_rotate)

        # 5) decode target (x,y) from action (no projection), compute z from current occupancy
        target_pos, (x, y, z) = self.get_target_position(action, box_size_after_rotate)

        # 6) place box & forward sim
        self.place_box(sample_box, target_pos, target_quat)
        self.sim_forward(40)

        # 7) stability check
        cur_pos = self.get_box_pose(sample_box_id)[:3]
        is_stable = self.check_stable() and (np.linalg.norm(cur_pos - target_pos) < self.stable_thres)

        if is_stable:
            # Commit placement into state
            self.unplaced_box_ids.pop(sample_box_index)
            self.boxes_on_pallet_id.append(sample_box_id)
            self.boxes_on_pallet_target_pose[sample_box_id] = np.concatenate([target_pos, target_quat])

            # Update pallet occupancy (density) observation
            X = int(self.pallet_size_discrete[0])
            Y = int(self.pallet_size_discrete[1])
            x2 = int(min(x + box_size_after_rotate[0], X))
            y2 = int(min(y + box_size_after_rotate[1], Y))
            z2 = int(z + int(box_size_after_rotate[2]))

            # fill region with density scalar (same as original code)
            self.obs["pallet_obs_density"][x:x2, y:y2, z:z2] = float(sample_box_density)

            # update buffer obs
            self.update_obs_buffer()

            done = (len(self.boxes_on_pallet_id) == self.total_box_number)
            termination_reason = 3 if done else 0
            reward, info = self.reward_func(termination_reason, box_size_after_rotate, (x, y, z))
        else:
            done = True
            reward, info = self.reward_func(termination_reason=2)

        if self.save_video:
            self.save_frame()

        info["record_data"] = record_data
        info["chosen_discrete"] = {"box_buffer_index": int(sample_box_index), "x": int(x), "y": int(y), "z": int(z)}
        info["box_density"] = float(sample_box_density)

        return self.obs, reward, done, info


class BoxPlanningEnvWrapper(gym.Env):
    """
    Optional Gym wrapper, still useful for:
    - standardized reset(seed=...)
    - simple rollouts / logging scripts
    Even if you don't use RL.
    """

    def __init__(self, save_video_path=None, device=torch.device("cuda:0")):
        super().__init__()
        self.env = BoxPlanning(
            save_video_path=save_video_path,
            device=device,
            init_box_pose_path="./helpers/box_init_pose.npy",
        )

        action_dim = int(self.env.N_visible_boxes + 6 + self.env.pallet_size_discrete[0] + self.env.pallet_size_discrete[1])
        action_lower = np.array([-10.0] * action_dim, dtype=np.float32)
        action_upper = np.array([10.0] * action_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=action_lower, high=action_upper)

        self.observation_space = gym.spaces.Dict(
            {
                "pallet_obs_density": spaces.Box(
                    low=0,
                    high=10,
                    shape=(int(self.env.pallet_size_discrete[0]), int(self.env.pallet_size_discrete[1]), self.env.max_pallet_height),
                ),
                "buffer": spaces.Box(low=0, high=10, shape=(self.env.N_visible_boxes * self.env.n_properties,)),
            }
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reinit(random_generator=self.np_random)
        return obs, {}

    # convenience passthrough
    @property
    def pallet_size_discrete(self):
        return self.env.pallet_size_discrete

    @property
    def N_visible_boxes(self):
        return self.env.N_visible_boxes


def encode_choice_to_action_logits(N, X, Y, box_i, rot_i, x_i, y_i, low=-10.0, high=10.0):
    """
    Helper (optional): convert a discrete choice into an action logits vector.
    """
    action_dim = int(N + 6 + X + Y)
    a = np.full(action_dim, low, dtype=np.float32)

    box_i = int(np.clip(box_i, 0, N - 1))
    rot_i = int(np.clip(rot_i, 0, 5))
    x_i = int(np.clip(x_i, 0, X - 1))
    y_i = int(np.clip(y_i, 0, Y - 1))

    a[box_i] = high
    a[N + rot_i] = high
    a[N + 6 + x_i] = high
    a[N + 6 + X + y_i] = high
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/debug.mp4", help="Path to save video")
    args = parser.parse_args()

    # Quick sanity check: random discrete choices -> logits -> env.step
    env = BoxPlanningEnvWrapper(save_video_path=args.video)
    obs, _ = env.reset(seed=0)

    N = env.N_visible_boxes
    X = int(env.pallet_size_discrete[0])
    Y = int(env.pallet_size_discrete[1])

    done = False
    step_count = 0
    while not done and step_count < 30:
        box_i = np.random.randint(0, N)
        rot_i = 0
        x_i = np.random.randint(0, X)
        y_i = np.random.randint(0, Y)

        action = encode_choice_to_action_logits(N, X, Y, box_i, rot_i, x_i, y_i)
        obs, reward, done, trunc, info = env.step(action)
        step_count += 1

    print("Finished. Steps:", step_count, "Done:", done)
