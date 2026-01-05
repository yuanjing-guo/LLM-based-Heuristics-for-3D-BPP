# env.py
# Pure-execution palletization environment (robosuite + gymnasium wrapper).
# - Heuristic outputs FINAL action logits (no mask inside env).
# - Env decodes logits -> choose box / rot / x / y -> computes z -> teleports box -> sim -> stability check -> state update.
# - STRICT: if any part of box would exceed pallet boundary -> raise RuntimeError.
#
# Reward (UPDATED):
#   - Use FINAL heightmap-envelope utilization as reward:
#       util = V_boxes / V_envelope_hm
#     where V_envelope_hm is computed from heightmap envelope of current pallet occupancy.
#   - Per-step reward = 0, only terminal reward = util (success or failure termination).

import copy
import os
import random
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import imageio
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


# ---------------------------
# Utilities
# ---------------------------

def resolve_video_path(path: Optional[str]) -> Optional[str]:
    """Append timestamp to a video path and ensure parent folder exists."""
    if path is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base, ext = os.path.splitext(path)
    ext = ext or ".mp4"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}__{timestamp}{ext}"


# 6 possible axis orders
# order is used like: rotated_size = size[order]
ORDERS = {
    0: np.array([0, 1, 2], dtype=int),
    1: np.array([0, 2, 1], dtype=int),
    2: np.array([1, 0, 2], dtype=int),
    3: np.array([1, 2, 0], dtype=int),
    4: np.array([2, 0, 1], dtype=int),
    5: np.array([2, 1, 0], dtype=int),
}


def quat_xyzw_from_order(order_id: int) -> np.ndarray:
    """
    Return target quaternion (xyzw) for the 6 discrete orientations.
    Matches your original compute_target_quat_from_order.
    """
    if order_id == 0:
        rotm = np.eye(3)
    elif order_id == 1:
        rotm = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    elif order_id == 2:
        rotm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    elif order_id == 3:
        rotm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    elif order_id == 4:
        rotm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    elif order_id == 5:
        rotm = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    else:
        raise ValueError(f"Invalid order_id: {order_id}")

    # robosuite T.mat2quat returns xyzw
    return T.mat2quat(rotm).astype(np.float32)


def compute_utilization_heightmap(pallet_obs_density: np.ndarray) -> Dict[str, float]:
    """
    Heightmap-envelope utilization (discrete bin world).

    pallet_obs_density shape: (X, Y, H)
      >0 means occupied (stored density value)
      =0 means empty

    Define:
      occ(x,y,z) = pallet_obs_density>0
      h(x,y) = highest occupied z + 1, else 0
      V_boxes = sum(occ)
      V_env_hm = sum(h)   # heightmap envelope volume
      util = V_boxes / V_env_hm

    Returns a dict of metrics (all in bins, except util).
    """
    if pallet_obs_density.ndim != 3:
        raise ValueError(f"pallet_obs_density must be (X,Y,H), got {pallet_obs_density.shape}")

    X, Y, H = pallet_obs_density.shape
    occ = pallet_obs_density > 0
    has = occ.any(axis=2)  # (X,Y) bool

    # heightmap: h(x,y) = max_z+1 if any occupied, else 0
    # compute "first occupied from top" using reversed z
    occ_rev = occ[..., ::-1]
    first_from_top = occ_rev.argmax(axis=2)  # (X,Y), valid even if all-False (returns 0)
    hmap = np.where(has, H - first_from_top, 0).astype(np.int32)  # (X,Y)

    V_boxes = float(occ.sum())               # bins^3
    V_env_hm = float(hmap.sum())             # bins^3
    util = float(V_boxes / max(V_env_hm, 1.0))

    footprint = float(has.sum())             # bins^2 (optional debug)
    hmax = float(hmap.max()) if hmap.size > 0 else 0.0

    return {
        "util": util,
        "V_boxes_bins3": V_boxes,
        "V_env_hm_bins3": V_env_hm,
        "footprint_bins2": footprint,
        "hmax_bins": hmax,
        "X": float(X),
        "Y": float(Y),
        "H": float(H),
    }


# ---------------------------
# Core Environment
# ---------------------------

class BoxPlanning(SingleArmEnv):
    """
    Pure-execution env:
    - action logits -> discrete (box_index, rot_id, x, y)
    - env computes z from pallet occupancy
    - teleports the box, runs physics, checks stability, updates state
    """

    def __init__(
        self,
        save_video_path: Optional[str] = None,
        init_box_pose_path: Optional[str] = None,
        control_freq: int = 20,
        horizon: int = 100,
        ignore_done: bool = True,
    ):
        # --- Table config ---
        self.table_full_size = TaskConfig.table.full_size
        self.table_friction = TaskConfig.table.friction
        self.table_offset = np.array(TaskConfig.table.offset, dtype=np.float32)

        # --- Pallet config ---
        self.pallet_size = np.array(TaskConfig.pallet.size, dtype=np.float32)
        self.pallet_position = self.table_offset + np.array(
            TaskConfig.pallet.relative_table_displacement, dtype=np.float32
        )

        # --- Task config ---
        self.N_visible_boxes = int(TaskConfig.buffer_size)
        self.max_pallet_height = int(TaskConfig.pallet.max_pallet_height)
        self.bin_size = float(TaskConfig.bin_size)
        self.pallet_size_discrete = (self.pallet_size[:2] / self.bin_size).astype(int)  # (X, Y)
        self.n_properties = int(TaskConfig.box.n_properties)  # e.g., 4: (sx, sy, sz, density)
        self.n_box_types = int(TaskConfig.box.n_type)

        # --- Stability threshold ---
        self.stable_thres = 0.02

        # --- Init box pose source (optional) ---
        self.init_box_pose_path = init_box_pose_path

        # --- Controller config ---
        controller_configs = load_controller_config(custom_fpath="./helpers/controller.json")

        # --- Video config ---
        self.save_video = save_video_path is not None
        self.writer = None
        if self.save_video:
            final_video_path = resolve_video_path(save_video_path)
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
    # Util logging helpers (NEW, additive only)
    # ---------------------------

    def _get_current_util_metrics(self) -> Dict[str, float]:
        """Compute util metrics from CURRENT committed pallet occupancy (obs['pallet_obs_density'])."""
        return compute_utilization_heightmap(self.obs["pallet_obs_density"])

    def _inject_current_util_into_info(self, info: Dict) -> Dict:
        """
        Add current util metrics into info without breaking original keys.
        We expose:
          - util_current
          - V_boxes_bins3, V_env_hm_bins3, footprint_bins2, hmax_bins
        """
        metrics = self._get_current_util_metrics()
        # keep original 'util' key if it already exists (terminal reward_func adds it)
        # but always provide util_current for per-step logging
        info.setdefault("V_boxes_bins3", metrics["V_boxes_bins3"])
        info.setdefault("V_env_hm_bins3", metrics["V_env_hm_bins3"])
        info.setdefault("footprint_bins2", metrics["footprint_bins2"])
        info.setdefault("hmax_bins", metrics["hmax_bins"])
        info["util_current"] = float(metrics["util"])
        return info

    def _print_step_util(self, info: Dict):
        """Per-step util print (requested)."""
        util = float(info.get("util_current", 0.0))
        vb = float(info.get("V_boxes_bins3", 0.0))
        ve = float(info.get("V_env_hm_bins3", 0.0))
        hm = float(info.get("hmax_bins", 0.0))
        fp = float(info.get("footprint_bins2", 0.0))
        term = int(info.get("termination_reason", -1))
        print(f"[UtilStep] util={util:.4f} V_boxes={vb:.0f} V_env_hm={ve:.0f} hmax={hm:.0f} footprint={fp:.0f} term={term}")

    # ---------------------------
    # Model construction
    # ---------------------------

    def create_box(self, box_type: int, box_name: str) -> BoxObject:
        """Create a box object with either rigid or soft physics parameters."""
        physics_cfg = getattr(TaskConfig, "physics", None)
        physics_mode = physics_cfg.mode if physics_cfg is not None else "soft"

        if physics_mode == "rigid":
            density = 10.0
            solref = [0.001, 1.0]
            solimp = [0.99, 0.99, 0.001]
            friction = (1.0, 0.005, 0.0001)
        elif physics_mode == "soft":
            cfg = TaskConfig.box.type_dict[box_type]
            density = cfg["density"]
            solref = [0.02, cfg["softness"]]
            solimp = [0.9, 0.95, 0.001]
            friction = cfg["friction"]
        else:
            raise ValueError(f"Unknown physics mode: {physics_mode}")

        return BoxObject(
            name=box_name,
            size=TaskConfig.box.type_dict[box_type]["size"],
            material=TaskConfig.box.type_dict[box_type]["material"],
            friction=friction,
            density=density,
            solref=solref,
            solimp=solimp,
        )

    def _load_model(self):
        super()._load_model()

        # Adjust robot base pose to table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Boxes
        if self.init_box_pose_path is None:
            self._load_boxes_from_recorded_pose()
        else:
            self._load_boxes_random_pose()

        # Pallet
        self.pallet = BoxObject(
            name="pallet",
            size=self.pallet_size / 2,
            material=TaskConfig.pallet.material,
        )
        self.pallet.get_obj().set("pos", array_to_string(self.pallet_position))

        # Assemble task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pallet] + [obj for sub in self.box_obj_list for obj in sub],
        )

    def _load_boxes_from_recorded_pose(self):
        """Load boxes and set their initial pose from a saved .npy file."""
        self.box_init_pose = np.load(self.init_box_pose_path).tolist()
        self.total_box_number = 0
        self.box_obj_list = [[] for _ in range(self.n_box_types)]

        for i in range(self.n_box_types):
            box_type = i + 1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.create_box(box_type, box_name)

                init_pose = np.array(self.box_init_pose[self.total_box_number], dtype=np.float32)
                box_obj.get_obj().set("pos", array_to_string(init_pose[:3]))
                box_obj.get_obj().set("quat", array_to_string(T.convert_quat(init_pose[3:], to="wxyz")))

                self.box_obj_list[i].append(box_obj)
                self.total_box_number += 1

    def _load_boxes_random_pose(self):
        """Load boxes and set random initial positions (roughly above the table)."""
        self.total_box_number = 0
        self.box_obj_list = [[] for _ in range(self.n_box_types)]

        for i in range(self.n_box_types):
            box_type = i + 1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.create_box(box_type, box_name)

                init_x = random.uniform(-0.3, 0.3)
                init_y = random.uniform(-0.8, -0.3)
                init_z = random.uniform(0, 0.05) + (self.n_box_types - 1 - i) * 0.05

                box_obj.get_obj().set(
                    "pos",
                    array_to_string(
                        self.table_offset - box_obj.bottom_offset
                        + np.array([init_x, init_y, init_z], dtype=np.float32)
                    ),
                )

                self.box_obj_list[i].append(box_obj)
                self.total_box_number += 1

    def _setup_references(self):
        super()._setup_references()

        self.boxes_ids = []
        self.id_to_box_obj: Dict[int, BoxObject] = {}
        self.id_to_properties: Dict[int, np.ndarray] = {}

        for i in range(self.n_box_types):
            for j in range(len(self.box_obj_list[i])):
                box_obj = self.box_obj_list[i][j]
                box_id = self.sim.model.body_name2id(box_obj.root_body)

                self.boxes_ids.append(box_id)
                self.id_to_box_obj[box_id] = box_obj

                # properties: (half_size * 100) + (density / 1000)
                props = list(np.array(box_obj.size, dtype=np.float32) * 100.0) + [float(box_obj.density) / 1000.0]
                self.id_to_properties[box_id] = np.array(props, dtype=np.float32)

    # ---------------------------
    # Action decoding
    # ---------------------------

    def choose_box_slot(self, action: np.ndarray) -> int:
        """Pick which buffer index to use from logits (argmax)."""
        return int(np.argmax(action[: self.N_visible_boxes]))

    def choose_rot_id(self, action: np.ndarray) -> int:
        """Pick one of 6 rotation IDs from logits (argmax)."""
        rot_logits = action[self.N_visible_boxes : self.N_visible_boxes + 6]
        return int(np.argmax(rot_logits))

    def get_target_position_strict(
        self,
        action: np.ndarray,
        box_size_after_rotate: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        STRICT: If any part of the box goes out of bounds -> raise RuntimeError.
        Decodes x,y from action logits and computes z from occupancy.
        """
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])

        x_logits = action[self.N_visible_boxes + 6 : self.N_visible_boxes + 6 + X]
        y_logits = action[self.N_visible_boxes + 6 + X : self.N_visible_boxes + 6 + X + Y]
        x = int(np.argmax(x_logits))
        y = int(np.argmax(y_logits))

        x = int(np.clip(x, 0, X - 1))
        y = int(np.clip(y, 0, Y - 1))

        dx = int(box_size_after_rotate[0])
        dy = int(box_size_after_rotate[1])

        # Correct strict bound:
        #   x+dx <= X, y+dy <= Y
        if (x + dx > X) or (y + dy > Y):
            raise RuntimeError(
                f"[OutOfBounds] Box would exceed pallet boundary: "
                f"(x,y)=({x},{y}), (dx,dy)=({dx},{dy}), pallet=(X,Y)=({X},{Y}). "
                f"Condition: x+dx<=X and y+dy<=Y must hold."
            )

        x2 = x + dx
        y2 = y + dy

        place_area = self.obs["pallet_obs_density"][x:x2, y:y2, :]
        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

        target_x = self.pallet_position[0] - self.pallet_size[0] / 2 + x * self.bin_size + dx * self.bin_size / 2
        target_y = self.pallet_position[1] - self.pallet_size[1] / 2 + y * self.bin_size + dy * self.bin_size / 2
        target_z = (
            self.pallet_position[2]
            + self.pallet_size[2] / 2
            + z * self.bin_size
            + int(box_size_after_rotate[2]) * self.bin_size / 2
        )

        target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)
        return target_pos, (x, y, z)

    # ---------------------------
    # Execution & state update
    # ---------------------------

    def place_box(self, box_obj: BoxObject, target_pos: np.ndarray, target_quat_xyzw: np.ndarray):
        """Teleport box to target pose. quat is xyzw."""
        self.sim.data.set_joint_qpos(
            box_obj.joints[0],
            np.concatenate([target_pos, T.convert_quat(target_quat_xyzw, to="wxyz")]),
        )
        self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

    def sim_forward(self, steps: int):
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def get_box_pose(self, box_id: int) -> np.ndarray:
        """Return [x,y,z,qx,qy,qz,qw] (quat in xyzw)."""
        box_pos = np.array(self.sim.data.body_xpos[box_id], dtype=np.float32)
        box_quat = convert_quat(np.array(self.sim.data.body_xquat[box_id], dtype=np.float32), to="xyzw")
        return np.hstack((box_pos, box_quat))

    def check_stable(self) -> bool:
        """Old boxes should stay near their target position."""
        for box_id in self.boxes_on_pallet_id:
            cur = self.get_box_pose(box_id)[:3]
            tgt = self.boxes_on_pallet_target_pose[box_id][:3]
            if np.linalg.norm(cur - tgt) > self.stable_thres:
                return False
        return True

    def update_obs_buffer(self) -> np.ndarray:
        """Fill obs['buffer'] with properties of first N unplaced boxes (zero-padded)."""
        buf = np.zeros(self.N_visible_boxes * self.n_properties, dtype=np.float32)
        n = min(len(self.unplaced_box_ids), self.N_visible_boxes)
        for i in range(n):
            props = self.id_to_properties[self.unplaced_box_ids[i]]
            buf[self.n_properties * i : self.n_properties * (i + 1)] = props
        self.obs["buffer"] = buf
        return buf

    def save_frame(self):
        """Render one frame and append to video."""
        if not self.save_video:
            return
        self._update_observables()
        img = self.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        self.writer.append_data(img)

    def record_pallet(self, box_id: int, rot_id: int, target_quat: np.ndarray, size_after_rotate: np.ndarray) -> Dict:
        """Create a debug record for offline analysis."""
        return {
            "pallet_config": copy.deepcopy(self.boxes_on_pallet_target_pose),
            "pallet_obs_density": self.obs["pallet_obs_density"].copy(),
            "to_place_id": int(box_id),
            "rot_id": int(rot_id),
            "to_place_quat": target_quat.copy(),  # xyzw
            "size_after_rotate": size_after_rotate.copy(),
            "to_place_density": float(self.id_to_properties[box_id][3]),
        }

    # ---------------------------
    # Reward (UPDATED)
    # ---------------------------

    def reward_func(self, termination_reason: int) -> Tuple[float, Dict]:
        """
        termination_reason:
            0 -> ongoing
            2 -> unstable (failed placement)
            3 -> success (placed all)
            4 -> invalid action (picked empty buffer slot)

        Reward policy (UPDATED):
          - ongoing: reward = 0
          - invalid action: reward = 0 (episode continues)
          - terminal (2 or 3): reward = final utilization (heightmap envelope)
        """
        info: Dict[str, float] = {"termination_reason": int(termination_reason)}

        if termination_reason in (2, 3):
            metrics = compute_utilization_heightmap(self.obs["pallet_obs_density"])
            r = float(metrics["util"])
            info.update(metrics)  # util, V_boxes_bins3, V_env_hm_bins3, footprint, hmax...
        else:
            r = 0.0

        info["reward_util_final"] = float(r)
        return float(r), info

    # ---------------------------
    # Reset / Reinit
    # ---------------------------

    def reinit(self, rng: np.random.Generator) -> Dict:
        """
        Deterministic reset used by wrapper.reset(seed=...).
        - If recorded init poses exist, teleport boxes back to their init pose.
        - Shuffle the order of unplaced boxes (affects buffer ordering).
        - Clear pallet occupancy.
        """
        self._reset_boxes_to_init_pose_if_available()

        self.unplaced_box_ids = copy.copy(self.boxes_ids)
        rng.shuffle(self.unplaced_box_ids)

        X, Y = int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1])
        self.obs = {
            "pallet_obs_density": np.zeros((X, Y, self.max_pallet_height), dtype=np.float32)
        }
        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}
        return self.obs

    def _reset_boxes_to_init_pose_if_available(self):
        """Teleport boxes to saved init pose if self.box_init_pose exists."""
        if not hasattr(self, "box_init_pose"):
            return
        for i in range(self.total_box_number):
            box_id = self.boxes_ids[i]
            init_pose = np.array(self.box_init_pose[i], dtype=np.float32)  # xyz + quat(xyzw)
            box_obj = self.id_to_box_obj[box_id]
            self.place_box(box_obj, init_pose[:3], init_pose[3:])

    # ---------------------------
    # Step
    # ---------------------------

    def step(self, action: np.ndarray):
        """
        Decode action -> place one box -> sim -> stable? -> update state.
        Returns: obs, reward, done, info   (wrapper converts to gymnasium 5-tuple)
        """
        # 1) decode box slot + rotation id
        box_slot = self.choose_box_slot(action)
        rot_id = self.choose_rot_id(action)

        # invalid: slot refers to empty padded buffer entry
        if box_slot >= len(self.unplaced_box_ids):
            reward, info = self.reward_func(termination_reason=4)
            done = False
            self.save_frame()
            info["record_data"] = None

            # NEW: add + print current util every step
            info = self._inject_current_util_into_info(info)
            self._print_step_util(info)

            return self.obs, reward, done, info

        # 2) resolve box id & properties
        box_id = self.unplaced_box_ids[box_slot]
        box_obj = self.id_to_box_obj[box_id]

        # discretized size (dx,dy,dz) in "natural axis order"
        box_size = (np.array(box_obj.size, dtype=np.float32) * 2.0 / self.bin_size).astype(int)
        box_density = float(self.id_to_properties[box_id][3])

        # 3) apply rotation: axis permutation + target quat
        order = ORDERS[int(np.clip(rot_id, 0, 5))]
        target_quat = quat_xyzw_from_order(rot_id)
        size_after_rotate = box_size[order]
        self._last_order_debug = order

        # 4) record state before placing
        record_data = self.record_pallet(box_id, rot_id, target_quat, size_after_rotate)

        # 5) decode x,y and compute z (strict bounds)
        target_pos, (x, y, z) = self.get_target_position_strict(action, size_after_rotate)

        print(
            f"[StepDebug] rot_id={int(rot_id)} "
            f"order={order.tolist()} "
            f"place_discrete=(x={int(x)}, y={int(y)}, z={int(z)}) "
            f"size_rot=(dx={int(size_after_rotate[0])}, dy={int(size_after_rotate[1])}, dz={int(size_after_rotate[2])})"
        )

        # 6) place & simulate
        self.place_box(box_obj, target_pos, target_quat)
        self.sim_forward(40)

        # 7) stability check
        cur_pos = self.get_box_pose(box_id)[:3]
        is_stable = self.check_stable() and (np.linalg.norm(cur_pos - target_pos) < self.stable_thres)

        if not is_stable:
            reward, info = self.reward_func(termination_reason=2)  # FINAL util on current committed pallet
            done = True
            self.save_frame()
            info["record_data"] = record_data
            info["chosen_discrete"] = {
                "box_buffer_index": int(box_slot),
                "rot_id": int(rot_id),
                "x": int(x),
                "y": int(y),
                "z": int(z),
            }
            info["box_density"] = float(box_density)

            # NEW: add + print current util every step (including terminal)
            info = self._inject_current_util_into_info(info)
            self._print_step_util(info)
            # NEW: final util print at terminal
            print(f"[FinalUtil] util={float(info.get('util_current', 0.0)):.4f}")

            return self.obs, reward, done, info

        # --- Commit placement ---
        self.unplaced_box_ids.pop(box_slot)
        self.boxes_on_pallet_id.append(box_id)
        self.boxes_on_pallet_target_pose[box_id] = np.concatenate([target_pos, target_quat])

        # Update occupancy (no clipping needed because strict bounds already checked)
        dx, dy, dz = int(size_after_rotate[0]), int(size_after_rotate[1]), int(size_after_rotate[2])
        x2, y2, z2 = x + dx, y + dy, z + dz
        self.obs["pallet_obs_density"][x:x2, y:y2, z:z2] = float(box_density)

        # Update buffer
        self.update_obs_buffer()

        # Episode termination
        done = (len(self.boxes_on_pallet_id) == self.total_box_number)
        term_reason = 3 if done else 0

        # Reward: only final util at terminal
        reward, info = self.reward_func(term_reason)

        # Video
        self.save_frame()

        # Info
        info["record_data"] = record_data
        info["chosen_discrete"] = {
            "box_buffer_index": int(box_slot),
            "rot_id": int(rot_id),
            "x": int(x),
            "y": int(y),
            "z": int(z),
        }
        info["box_density"] = float(box_density)

        # NEW: add + print current util every step
        info = self._inject_current_util_into_info(info)
        self._print_step_util(info)

        # NEW: final util print at success terminal
        if term_reason == 3:
            print(f"[FinalUtil] util={float(info.get('util_current', 0.0)):.4f}")

        return self.obs, reward, done, info


# ---------------------------
# Gymnasium Wrapper
# ---------------------------

class BoxPlanningEnvWrapper(gym.Env):
    """Gymnasium wrapper to standardize reset(seed=...) and step(...) signature."""

    metadata = {"render_modes": []}

    def __init__(self, save_video_path: Optional[str] = None):
        super().__init__()
        self.env = BoxPlanning(
            save_video_path=save_video_path,
            init_box_pose_path="./helpers/box_init_pose.npy",
        )

        X, Y = int(self.env.pallet_size_discrete[0]), int(self.env.pallet_size_discrete[1])
        N = int(self.env.N_visible_boxes)

        action_dim = int(N + 6 + X + Y)
        self.action_space = spaces.Box(
            low=np.full(action_dim, -10.0, dtype=np.float32),
            high=np.full(action_dim, 10.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "pallet_obs_density": spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(X, Y, self.env.max_pallet_height),
                    dtype=np.float32,
                ),
                "buffer": spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(N * self.env.n_properties,),
                    dtype=np.float32,
                ),
            }
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reinit(rng=self.np_random)
        return obs, {}

    @property
    def pallet_size_discrete(self):
        return self.env.pallet_size_discrete

    @property
    def N_visible_boxes(self):
        return self.env.N_visible_boxes


# ---------------------------
# Helpers
# ---------------------------

def encode_choice_to_action_logits(
    N: int, X: int, Y: int,
    box_i: int, rot_i: int, x_i: int, y_i: int,
    low: float = -10.0, high: float = 10.0
) -> np.ndarray:
    """
    Convert discrete choice into an action logits vector (one-hot style).
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


def decode_box_discrete_size_from_buffer(
    obs: Dict, box_slot: int, n_properties: int, bin_size: float
) -> np.ndarray:
    """
    Read size (dx,dy,dz) from obs['buffer'] for the box in a given slot.
    buffer stores half-size*100, so:
        half_size_m = value / 100
        full_size_m = half_size_m * 2
        d = int(full_size_m / bin_size)
    Returns size in natural axis order (x,y,z).
    """
    buf = obs["buffer"]
    base = box_slot * n_properties
    half_sizes_m = (buf[base : base + 3] / 100.0).astype(np.float32)
    full_sizes_m = half_sizes_m * 2.0
    dxyz = (full_sizes_m / bin_size).astype(int)
    return dxyz


# ---------------------------
# Sanity Check (safe sampling under STRICT bounds)
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/debug.mp4", help="Path to save video")
    args = parser.parse_args()

    env = BoxPlanningEnvWrapper(save_video_path=args.video)
    obs, _ = env.reset(seed=0)

    N = env.N_visible_boxes
    X = int(env.pallet_size_discrete[0])
    Y = int(env.pallet_size_discrete[1])

    rng = np.random.default_rng(0)

    done = False
    step_count = 0

    while not done and step_count < 30:
        # Choose a valid slot only (avoid padded empty slots)
        remaining = len(env.env.unplaced_box_ids)
        slot_max = max(1, min(remaining, N))
        box_i = int(rng.integers(0, slot_max))

        # Random rotation id (0..5)
        rot_i = int(rng.integers(0, 6))

        # Compute rotated (dx,dy) for strict in-bounds sampling
        dxyz = decode_box_discrete_size_from_buffer(
            obs, box_i, n_properties=env.env.n_properties, bin_size=env.env.bin_size
        )
        order = ORDERS[rot_i]
        dxyz_rot = dxyz[order]
        dx, dy = int(dxyz_rot[0]), int(dxyz_rot[1])

        max_x = max(0, X - dx)
        max_y = max(0, Y - dy)
        x_i = int(rng.integers(0, max_x + 1))
        y_i = int(rng.integers(0, max_y + 1))

        action = encode_choice_to_action_logits(N, X, Y, box_i, rot_i, x_i, y_i)
        obs, reward, done, trunc, info = env.step(action)
        step_count += 1

    # final metrics
    metrics = compute_utilization_heightmap(env.env.obs["pallet_obs_density"])
    print("Finished. Steps:", step_count, "Done:", done)
    print(
        "[FinalUtil]",
        f"util={metrics['util']:.4f}",
        f"V_boxes={metrics['V_boxes_bins3']:.0f}",
        f"V_env_hm={metrics['V_env_hm_bins3']:.0f}",
        f"hmax={metrics['hmax_bins']:.0f}",
        f"footprint={metrics['footprint_bins2']:.0f}",
    )
