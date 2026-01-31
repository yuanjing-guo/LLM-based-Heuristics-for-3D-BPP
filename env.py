# env.py
# Pure-execution palletization environment (robosuite + gymnasium wrapper).
# - Heuristic outputs FINAL action logits (no mask inside env).
# - Env decodes logits -> choose box / rot / x / y -> computes z -> teleports box -> sim -> stability check -> state update.
# - STRICT: if any part of box would exceed pallet boundary -> raise RuntimeError (XY only).
# - HEIGHT policy (UPDATED): if box would exceed max height -> TERMINATE episode gracefully (no crash), return util in info.
#
# Reward (UPDATED):
#   - Use FINAL VOLUME utilization as terminal reward:
#       util = V_boxes / V_bin
#     where V_boxes is occupied voxel count, V_bin = X*Y*H (discrete bin volume).
#   - Per-step reward = 0, only terminal reward = util (success / unstable / height_oob termination).
#
# Logging (UPDATED):
#   - Each step prints ONLY: placement discrete (x,y,z) and rot_id + chosen buffer slot
#   - Episode end prints ONCE: util and number of committed (stable) boxes
#
# Physics-aware obs (NEW, backward compatible):
#   - Keep obs["buffer"] unchanged: size(3) + density(1)  (TaskConfig.box.n_properties=4)
#   - Add obs["buffer_physics"]: per-slot [softness, friction_mu]
#   - Add obs["pallet_obs_softness"]: voxel field aligned with pallet_obs_density
#
# IMPORTANT POLICY (UPDATED):
#   - TaskConfig no longer decides physics mode.
#   - ONLY when physics_mode == "soft" we enable soft contact.
#   - Otherwise (None / "rigid") => rigid mode by default.

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


class HeightLimitExceeded(RuntimeError):
    """
    Graceful (non-crashing) terminal condition:
    placement would exceed max height.
    Carries discrete placement info for logging/debug.
    """
    def __init__(self, msg: str, *, x: int, y: int, z: int, dz: int, H: int):
        super().__init__(msg)
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.dz = int(dz)
        self.H = int(H)


def quat_xyzw_from_order(order_id: int) -> np.ndarray:
    """Return target quaternion (xyzw) for the 6 discrete orientations."""
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


def compute_utilization_volume(pallet_obs_density: np.ndarray) -> Dict[str, float]:
    """Volume utilization util = occupied_voxels / (X*Y*H)."""
    if pallet_obs_density.ndim != 3:
        raise ValueError(f"pallet_obs_density must be (X,Y,H), got {pallet_obs_density.shape}")

    X, Y, H = pallet_obs_density.shape
    occ = pallet_obs_density > 0

    V_boxes = float(occ.sum())
    V_bin = float(X * Y * H)
    util = float(V_boxes / max(V_bin, 1.0))

    return {
        "util": util,
        "V_boxes_bins3": V_boxes,
        "V_bin_bins3": V_bin,
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

    Physics-aware obs:
      - obs["pallet_obs_softness"] : voxel field (X,Y,H)
      - obs["buffer_physics"]      : per-slot [softness, friction_mu]
    """

    def __init__(
        self,
        save_video_path: Optional[str] = None,
        init_box_pose_path: Optional[str] = None,
        control_freq: int = 20,
        horizon: int = 100,
        ignore_done: bool = True,
        physics_mode: Optional[str] = None,   # "soft" enables soft contacts; otherwise rigid default
        expose_physics_obs: bool = True,
    ):
        # ------------------------------------------------------------
        # Physics mode policy (UPDATED):
        #   - ONLY physics_mode == "soft" => soft contact
        #   - otherwise => rigid
        # ------------------------------------------------------------
        if physics_mode is None:
            self.physics_mode = "rigid"
        else:
            pm = str(physics_mode).lower().strip()
            if pm == "soft":
                self.physics_mode = "soft"
            elif pm == "rigid":
                self.physics_mode = "rigid"
            else:
                raise ValueError(f"Invalid physics_mode={physics_mode}. Use 'soft' or 'rigid' (or omit for rigid).")

        self.expose_physics_obs = bool(expose_physics_obs)

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
        self.n_properties = int(TaskConfig.box.n_properties)  # keep as-is (e.g., 4)
        self.n_box_types = int(TaskConfig.box.n_type)

        # --- NEW: buffer_physics dims ---
        self.n_physics_properties = 2  # [softness, friction_mu]

        # --- Stability threshold ---
        self.stable_thres = 0.01

        # --- Early stop controls (NEW) ---
        # If a placement becomes unstable during sim, stop sim immediately and terminate episode.
        self.early_stop_on_unstable = True
        self.early_stop_check_every = 1  # check stability every N sim steps

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
    # Model construction
    # ---------------------------

    def create_box(self, box_type: int, box_name: str) -> BoxObject:
        """
        Create a box object with either rigid or soft contact parameters.

        Design choice:
          - density/friction always from TaskConfig (so "weight" differences remain meaningful)
          - physics_mode mainly changes contact softness (solref/solimp)
        """
        cfg = TaskConfig.box.type_dict[box_type]
        density = cfg["density"]
        friction = cfg["friction"]
        softness = float(cfg.get("softness", 1.0))

        if self.physics_mode == "rigid":
            solref = [0.0001, 1.0]
            solimp = [0.999, 0.999, 0.00001]
        elif self.physics_mode == "soft":
            solref = [0.03, softness]
            solimp = [0.9, 0.95, 0.001]
        else:
            raise ValueError(f"Unknown physics_mode: {self.physics_mode}")

        return BoxObject(
            name=box_name,
            size=cfg["size"],
            material=cfg["material"],
            friction=friction,
            density=density,
            solref=solref,
            solimp=solimp,
        )

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])



        # Boxes
        if self.init_box_pose_path is not None:
        
            self._load_boxes_from_recorded_pose()
        else:
            self._load_boxes_random_pose()

        self.pallet = BoxObject(
            name="pallet",
            size=self.pallet_size / 2,
            material=TaskConfig.pallet.material,
        )
        self.pallet.get_obj().set("pos", array_to_string(self.pallet_position))

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pallet] + [obj for sub in self.box_obj_list for obj in sub],
        )

    def _load_boxes_from_recorded_pose(self):
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

        # original: size(3) + density(1) [scaled]
        self.id_to_properties: Dict[int, np.ndarray] = {}

        # physics properties for obs
        self.id_to_softness: Dict[int, float] = {}
        self.id_to_friction_mu: Dict[int, float] = {}

        for i in range(self.n_box_types):
            box_type = i + 1
            cfg = TaskConfig.box.type_dict[box_type]

            softness_val = float(cfg.get("softness", 1.0))
            mu_val = float(cfg.get("friction", (1.0, 0.005, 0.0001))[0])

            for j in range(len(self.box_obj_list[i])):
                box_obj = self.box_obj_list[i][j]
                box_id = self.sim.model.body_name2id(box_obj.root_body)

                self.boxes_ids.append(box_id)
                self.id_to_box_obj[box_id] = box_obj

                # properties: (half_size * 100) + (density / 1000)
                props = list(np.array(box_obj.size, dtype=np.float32) * 100.0) + [float(box_obj.density) / 1000.0]
                self.id_to_properties[box_id] = np.array(props, dtype=np.float32)

                self.id_to_softness[box_id] = softness_val
                self.id_to_friction_mu[box_id] = mu_val

    # ---------------------------
    # Action decoding
    # ---------------------------

    def choose_box_slot(self, action: np.ndarray) -> int:
        return int(np.argmax(action[: self.N_visible_boxes]))

    def choose_rot_id(self, action: np.ndarray) -> int:
        rot_logits = action[self.N_visible_boxes : self.N_visible_boxes + 6]
        return int(np.argmax(rot_logits))

    def get_target_position_strict(
        self,
        action: np.ndarray,
        box_size_after_rotate: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])
        H = int(self.max_pallet_height)

        x_logits = action[self.N_visible_boxes + 6 : self.N_visible_boxes + 6 + X]
        y_logits = action[self.N_visible_boxes + 6 + X : self.N_visible_boxes + 6 + X + Y]
        x = int(np.argmax(x_logits))
        y = int(np.argmax(y_logits))

        x = int(np.clip(x, 0, X - 1))
        y = int(np.clip(y, 0, Y - 1))

        dx = int(box_size_after_rotate[0])
        dy = int(box_size_after_rotate[1])
        dz = int(box_size_after_rotate[2])

        # XY strict boundary -> still a hard RuntimeError
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

        # Z strict height -> graceful terminal (NO CRASH)
        if (z + dz > H):
            raise HeightLimitExceeded(
                f"[OutOfBoundsZ] Box would exceed max height: "
                f"(x,y)=({x},{y}), (z,dz)=({z},{dz}), H={H}. Need z+dz<=H.",
                x=x, y=y, z=z, dz=dz, H=H
            )

        target_x = self.pallet_position[0] - self.pallet_size[0] / 2 + x * self.bin_size + dx * self.bin_size / 2
        target_y = self.pallet_position[1] - self.pallet_size[1] / 2 + y * self.bin_size + dy * self.bin_size / 2
        target_z = (
            self.pallet_position[2]
            + self.pallet_size[2] / 2
            + z * self.bin_size
            + dz * self.bin_size / 2
        )

        target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)
        return target_pos, (x, y, z)

    # ---------------------------
    # Execution & state update
    # ---------------------------

    def place_box(self, box_obj: BoxObject, target_pos: np.ndarray, target_quat_xyzw: np.ndarray):
        self.sim.data.set_joint_qpos(
            box_obj.joints[0],
            np.concatenate([target_pos, T.convert_quat(target_quat_xyzw, to="wxyz")]),
        )
        self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

    def sim_forward(self, steps: int):
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def sim_forward_until_unstable(
        self,
        steps: int,
        placed_box_id: int,
        placed_target_pos: np.ndarray,
        check_every: int = 1,
    ) -> bool:
        """
        Run physics for at most `steps`. If any already-committed box (or the newly placed box)
        deviates beyond stable_thres, stop immediately and return False.
        Return True if stayed stable for all steps.
        """
        check_every = max(1, int(check_every))
        for k in range(steps):
            self.sim.forward()
            self.sim.step()

            if (k + 1) % check_every != 0:
                continue

            # committed boxes must remain near their target pose
            if not self.check_stable():
                return False

            # newly placed box must remain near its intended target position
            cur_pos = self.get_box_pose(placed_box_id)[:3]
            if np.linalg.norm(cur_pos - placed_target_pos) > self.stable_thres:
                return False

        return True

    def get_box_pose(self, box_id: int) -> np.ndarray:
        box_pos = np.array(self.sim.data.body_xpos[box_id], dtype=np.float32)
        box_quat = convert_quat(np.array(self.sim.data.body_xquat[box_id], dtype=np.float32), to="xyzw")
        return np.hstack((box_pos, box_quat))

    def check_stable(self) -> bool:
        for box_id in self.boxes_on_pallet_id:
            cur = self.get_box_pose(box_id)[:3]
            tgt = self.boxes_on_pallet_target_pose[box_id][:3]
            if np.linalg.norm(cur - tgt) > self.stable_thres:
                return False
        return True

    def update_obs_buffer(self) -> None:
        """
        Fill obs['buffer'] with properties of first N unplaced boxes (zero-padded).
        Also fill obs['buffer_physics'].
        """
        buf = np.zeros(self.N_visible_boxes * self.n_properties, dtype=np.float32)
        buf_phy = np.zeros(self.N_visible_boxes * self.n_physics_properties, dtype=np.float32)

        n = min(len(self.unplaced_box_ids), self.N_visible_boxes)
        for i in range(n):
            box_id = self.unplaced_box_ids[i]

            props = self.id_to_properties[box_id]
            buf[self.n_properties * i : self.n_properties * (i + 1)] = props

            if self.expose_physics_obs:
                softness = float(self.id_to_softness.get(box_id, 0.0))
                mu = float(self.id_to_friction_mu.get(box_id, 0.0))
                base = self.n_physics_properties * i
                buf_phy[base + 0] = softness
                buf_phy[base + 1] = mu

        self.obs["buffer"] = buf
        self.obs["buffer_physics"] = buf_phy

    def save_frame(self):
        if not self.save_video:
            return
        self._update_observables()
        img = self.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        self.writer.append_data(img)

    def record_pallet(self, box_id: int, rot_id: int, target_quat: np.ndarray, size_after_rotate: np.ndarray) -> Dict:
        rec = {
            "pallet_config": copy.deepcopy(self.boxes_on_pallet_target_pose),
            "pallet_obs_density": self.obs["pallet_obs_density"].copy(),
            "to_place_id": int(box_id),
            "rot_id": int(rot_id),
            "to_place_quat": target_quat.copy(),  # xyzw
            "size_after_rotate": size_after_rotate.copy(),
            "to_place_density": float(self.id_to_properties[box_id][3]),
        }
        if "pallet_obs_softness" in self.obs:
            rec["pallet_obs_softness"] = self.obs["pallet_obs_softness"].copy()
            rec["to_place_softness"] = float(self.id_to_softness.get(box_id, 0.0))
        return rec

    # ---------------------------
    # Reward
    # ---------------------------

    def reward_func(self, termination_reason: int) -> Tuple[float, Dict]:
        info: Dict[str, float] = {
            "termination_reason": int(termination_reason),
            "physics_mode": str(self.physics_mode),
            "expose_physics_obs": int(self.expose_physics_obs),
        }

        # terminal util on: unstable(2), success(3), height_oob(5)
        if termination_reason in (2, 3, 5):
            metrics = compute_utilization_volume(self.obs["pallet_obs_density"])
            r = float(metrics["util"])
            info.update(metrics)
        else:
            r = 0.0

        info["reward_util_final"] = float(r)
        return float(r), info

    # ---------------------------
    # Reset / Reinit
    # ---------------------------

    def reinit(self, rng: np.random.Generator) -> Dict:
        self._reset_boxes_to_init_pose_if_available()

        self.unplaced_box_ids = copy.copy(self.boxes_ids)
        rng.shuffle(self.unplaced_box_ids)

        X, Y = int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1])

        self.obs = {
            "pallet_obs_density": np.zeros((X, Y, self.max_pallet_height), dtype=np.float32),
            "pallet_obs_softness": np.zeros((X, Y, self.max_pallet_height), dtype=np.float32),
            "buffer": np.zeros((self.N_visible_boxes * self.n_properties,), dtype=np.float32),
            "buffer_physics": np.zeros((self.N_visible_boxes * self.n_physics_properties,), dtype=np.float32),
        }

        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}
        return self.obs

    def _reset_boxes_to_init_pose_if_available(self):
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
        box_slot = self.choose_box_slot(action)
        rot_id = self.choose_rot_id(action)

        if box_slot >= len(self.unplaced_box_ids):
            reward, info = self.reward_func(termination_reason=4)
            done = False
            self.save_frame()
            info["record_data"] = None
            return self.obs, reward, done, info

        box_id = self.unplaced_box_ids[box_slot]
        box_obj = self.id_to_box_obj[box_id]

        box_size = (np.array(box_obj.size, dtype=np.float32) * 2.0 / self.bin_size).astype(int)
        box_density = float(self.id_to_properties[box_id][3])  # density/1000
        box_softness = float(self.id_to_softness.get(box_id, 0.0))

        order = ORDERS[int(np.clip(rot_id, 0, 5))]
        target_quat = quat_xyzw_from_order(rot_id)
        size_after_rotate = box_size[order]

        record_data = self.record_pallet(box_id, rot_id, target_quat, size_after_rotate)

        # --------- UPDATED: height_oob -> graceful done (no crash) ---------
        try:
            target_pos, (x, y, z) = self.get_target_position_strict(action, size_after_rotate)
        except HeightLimitExceeded as e:
            reward, info = self.reward_func(termination_reason=5)  # 5: height_oob
            done = True
            self.save_frame()

            info["error"] = str(e)
            info["record_data"] = record_data
            info["chosen_discrete"] = {
                "box_buffer_index": int(box_slot),
                "rot_id": int(rot_id),
                "x": int(e.x),
                "y": int(e.y),
                "z": int(e.z),
            }
            info["box_density"] = float(box_density)
            info["box_softness"] = float(box_softness)

            final_util = float(info.get("util", 0.0))
            n_boxes = int(len(self.boxes_on_pallet_id))
            print(f"[EpisodeEnd] reason=height_oob util={final_util:.4f} n_boxes={n_boxes}")
            return self.obs, reward, done, info

        print(f"[Step] slot={int(box_slot)} rot_id={int(rot_id)} place=(x={int(x)}, y={int(y)}, z={int(z)})")

        self.place_box(box_obj, target_pos, target_quat)

        # --------- early stop inside physics rollout ---------
        if self.early_stop_on_unstable:
            is_stable = self.sim_forward_until_unstable(
                steps=40,
                placed_box_id=box_id,
                placed_target_pos=target_pos,
                check_every=self.early_stop_check_every,
            )
        else:
            self.sim_forward(40)
            cur_pos = self.get_box_pose(box_id)[:3]
            is_stable = self.check_stable() and (np.linalg.norm(cur_pos - target_pos) < self.stable_thres)

        if not is_stable:
            reward, info = self.reward_func(termination_reason=2)
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
            info["box_softness"] = float(box_softness)

            final_util = float(info.get("util", 0.0))
            n_boxes = int(len(self.boxes_on_pallet_id))
            print(f"[EpisodeEnd] reason=unstable util={final_util:.4f} n_boxes={n_boxes}")

            return self.obs, reward, done, info

        # Commit placement
        self.unplaced_box_ids.pop(box_slot)
        self.boxes_on_pallet_id.append(box_id)
        self.boxes_on_pallet_target_pose[box_id] = np.concatenate([target_pos, target_quat])

        dx, dy, dz = int(size_after_rotate[0]), int(size_after_rotate[1]), int(size_after_rotate[2])
        x2, y2, z2 = x + dx, y + dy, z + dz

        self.obs["pallet_obs_density"][x:x2, y:y2, z:z2] = float(box_density)
        if self.expose_physics_obs:
            self.obs["pallet_obs_softness"][x:x2, y:y2, z:z2] = float(box_softness)

        self.update_obs_buffer()

        done = (len(self.boxes_on_pallet_id) == self.total_box_number)
        term_reason = 3 if done else 0

        reward, info = self.reward_func(term_reason)
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
        info["box_softness"] = float(box_softness)

        if term_reason == 3:
            final_util = float(info.get("util", 0.0))
            n_boxes = int(len(self.boxes_on_pallet_id))
            print(f"[EpisodeEnd] reason=success util={final_util:.4f} n_boxes={n_boxes}")

        return self.obs, reward, done, info


# ---------------------------
# Gymnasium Wrapper
# ---------------------------

class BoxPlanningEnvWrapper(gym.Env):
    """Gymnasium wrapper to standardize reset(seed=...) and step(...) signature."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        save_video_path: Optional[str] = None,
        physics_mode: Optional[str] = None,   # None => rigid default; "soft" => soft; "rigid" => rigid
        expose_physics_obs: bool = True,
    ):
        super().__init__()
        self.env = BoxPlanning(
            save_video_path=save_video_path,
            init_box_pose_path="./helpers/box_init_pose.npy",
            physics_mode=physics_mode,
            expose_physics_obs=expose_physics_obs,
        )

        X, Y = int(self.env.pallet_size_discrete[0]), int(self.env.pallet_size_discrete[1])
        H = int(self.env.max_pallet_height)
        N = int(self.env.N_visible_boxes)

        action_dim = int(N + 6 + X + Y)
        self.action_space = spaces.Box(
            low=np.full(action_dim, -10.0, dtype=np.float32),
            high=np.full(action_dim, 10.0, dtype=np.float32),
            dtype=np.float32,
        )

        obs_spaces = {
            "pallet_obs_density": spaces.Box(
                low=0.0,
                high=10.0,
                shape=(X, Y, H),
                dtype=np.float32,
            ),
            "buffer": spaces.Box(
                low=0.0,
                high=10.0,
                shape=(N * self.env.n_properties,),
                dtype=np.float32,
            ),
            "pallet_obs_softness": spaces.Box(
                low=0.0,
                high=10.0,
                shape=(X, Y, H),
                dtype=np.float32,
            ),
            "buffer_physics": spaces.Box(
                low=0.0,
                high=10.0,
                shape=(N * self.env.n_physics_properties,),
                dtype=np.float32,
            ),
        }
        self.observation_space = spaces.Dict(obs_spaces)

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
    buf = obs["buffer"]
    base = box_slot * n_properties
    half_sizes_m = (buf[base : base + 3] / 100.0).astype(np.float32)
    full_sizes_m = half_sizes_m * 2.0
    dxyz = (full_sizes_m / bin_size).astype(int)
    return dxyz


# ---------------------------
# Sanity Check
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/debug.mp4", help="Path to save video")

    # UPDATED: default rigid; only --physics_mode soft enables soft contacts
    parser.add_argument(
        "--physics_mode",
        type=str,
        default=None,                 # None => rigid
        choices=["rigid", "soft"],
        help="Physics contact mode. Omit => rigid. Use 'soft' to enable soft contacts.",
    )
    parser.add_argument("--no_physics_obs", action="store_true", help="Disable physics-aware obs filling")
    args = parser.parse_args()

    env = BoxPlanningEnvWrapper(
        save_video_path=args.video,
        physics_mode=args.physics_mode,
        expose_physics_obs=(not args.no_physics_obs),
    )
    obs, _ = env.reset(seed=0)

    N = env.N_visible_boxes
    X = int(env.pallet_size_discrete[0])
    Y = int(env.pallet_size_discrete[1])

    rng = np.random.default_rng(0)

    done = False
    step_count = 0

    while not done and step_count < 30:
        remaining = len(env.env.unplaced_box_ids)
        slot_max = max(1, min(remaining, N))
        box_i = int(rng.integers(0, slot_max))
        rot_i = int(rng.integers(0, 6))

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

    metrics = compute_utilization_volume(env.env.obs["pallet_obs_density"])
    print("Finished. Steps:", step_count, "Done:", done)
    print(
        "[FinalUtil]",
        f"util={metrics['util']:.4f}",
        f"V_boxes={metrics['V_boxes_bins3']:.0f}",
        f"V_bin={metrics['V_bin_bins3']:.0f}",
    )
