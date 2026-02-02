# envs/mixed_env.py
# Mixed packing + unpacking environment (robosuite + gymnasium wrapper)
# - Action is int32 array of length 5:
#     op=0 (PLACE):  [0, slot, rot_id, x, y]
#     op=1 (REMOVE): [1, pallet_index, 0, 0, 0]
#
# - Two-level candidate pools:
#     buffer_front (N): visible candidates -> obs["buffer"] filled from these ids
#     backlog_queue (big): remaining unseen boxes
#     rehandle_pool (big): removed boxes, can be reintroduced later
#     cooldown: removed boxes wait K steps before re-entering buffer_front
#
# - IMPORTANT: maintain pallet_footprints aligned with boxes_on_pallet_id
#   and expose:
#     obs["pallet_count"]
#     obs["pallet_footprints"]

import copy
import os
import random
from datetime import datetime
from typing import Dict, Tuple, Optional, List

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

# ✅ NEW: centralized removal mask checker (obs exposure only)
from heuristics.feasibility import RemovalFeasibilityChecker


# ---------------------------
# Utilities
# ---------------------------

def resolve_video_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base, ext = os.path.splitext(path)
    ext = ext or ".mp4"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}__{timestamp}{ext}"


# 6 possible axis orders
ORDERS = {
    0: np.array([0, 1, 2], dtype=int),
    1: np.array([0, 2, 1], dtype=int),
    2: np.array([1, 0, 2], dtype=int),
    3: np.array([1, 2, 0], dtype=int),
    4: np.array([2, 0, 1], dtype=int),
    5: np.array([2, 1, 0], dtype=int),
}


class HeightLimitExceeded(RuntimeError):
    def __init__(self, msg: str, *, x: int, y: int, z: int, dz: int, H: int):
        super().__init__(msg)
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.dz = int(dz)
        self.H = int(H)


def quat_xyzw_from_order(order_id: int) -> np.ndarray:
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
    return T.mat2quat(rotm).astype(np.float32)


def compute_utilization_volume(pallet_obs_density: np.ndarray) -> Dict[str, float]:
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
# Core Mixed Environment
# ---------------------------

class MixedBoxPlanning(SingleArmEnv):
    """
    Mixed env with PLACE and REMOVE.
    Maintains pallet footprints, exposes them to obs.
    """

    def __init__(
        self,
        save_video_path: Optional[str] = None,
        init_box_pose_path: Optional[str] = None,
        control_freq: int = 20,
        horizon: int = 200,
        ignore_done: bool = True,
        physics_mode: Optional[str] = None,   # "soft" enables soft contacts; otherwise rigid default
        expose_physics_obs: bool = True,
        # candidate scheduling
        rehandle_cooldown_steps: int = 5,
        settle_steps_after_place: int = 10, 
        settle_steps_after_remove: int = 10,
    ):
        # physics mode
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
        self.rehandle_cooldown_steps = int(rehandle_cooldown_steps)
        self.settle_steps_after_place = int(settle_steps_after_place)
        self.settle_steps_after_remove = int(settle_steps_after_remove)
        # table / pallet config
        self.table_full_size = TaskConfig.table.full_size
        self.table_friction = TaskConfig.table.friction
        self.table_offset = np.array(TaskConfig.table.offset, dtype=np.float32)

        self.pallet_size = np.array(TaskConfig.pallet.size, dtype=np.float32)
        self.pallet_position = self.table_offset + np.array(
            TaskConfig.pallet.relative_table_displacement, dtype=np.float32
        )

        # task config
        self.N_visible_boxes = int(TaskConfig.buffer_size)
        self.max_pallet_height = int(TaskConfig.pallet.max_pallet_height)
        self.bin_size = float(TaskConfig.bin_size)
        self.pallet_size_discrete = (self.pallet_size[:2] / self.bin_size).astype(int)  # (X, Y)
        self.n_properties = int(TaskConfig.box.n_properties)
        self.n_box_types = int(TaskConfig.box.n_type)

        self.n_physics_properties = 2  # [softness, friction_mu]

        # ---------------------------
        # Stability: STRICT (match env.py)
        # ---------------------------
        self.stable_thres = 0.01
        self.early_stop_on_unstable = True
        self.early_stop_check_every = 1

        # ✅ NEW: removal checker (for obs exposure only)
        self.remove_checker = RemovalFeasibilityChecker(H=self.max_pallet_height)

        # init pose
        self.init_box_pose_path = init_box_pose_path

        # controller config
        controller_configs = load_controller_config(custom_fpath="./helpers/controller.json")

        # video
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
    def _settle(self, steps: int = 10):
        steps = int(max(0, steps))
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def create_box(self, box_type: int, box_name: str) -> BoxObject:
        cfg = TaskConfig.box.type_dict[box_type]
        density = cfg["density"]
        friction = cfg["friction"]
        softness = float(cfg.get("softness", 1.0))

        if self.physics_mode == "rigid":
            solref = [0.0001, 1.0]
            solimp = [0.999, 0.999, 0.00001]
        else:
            solref = [0.01, softness]
            solimp = [0.97, 0.99, 0.001]

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

                init_x = random.uniform(-0.7, 0.7)
                init_y = random.uniform(-0.7, -0.3)
                init_z = random.uniform(0, 0.2) + (self.n_box_types - 1 - i) * 0.05

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

                props = list(np.array(box_obj.size, dtype=np.float32) * 100.0) + [float(box_obj.density) / 1000.0]
                self.id_to_properties[box_id] = np.array(props, dtype=np.float32)

                self.id_to_softness[box_id] = softness_val
                self.id_to_friction_mu[box_id] = mu_val

    # ---------------------------
    # Placement helpers
    # ---------------------------

    def place_box(self, box_obj: BoxObject, target_pos: np.ndarray, target_quat_xyzw: np.ndarray):
        self.sim.data.set_joint_qpos(
            box_obj.joints[0],
            np.concatenate([target_pos, T.convert_quat(target_quat_xyzw, to="wxyz")]),
        )
        self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

    def get_box_pose(self, box_id: int) -> np.ndarray:
        box_pos = np.array(self.sim.data.body_xpos[box_id], dtype=np.float32)
        box_quat = convert_quat(np.array(self.sim.data.body_xquat[box_id], dtype=np.float32), to="xyzw")
        return np.hstack((box_pos, box_quat))

    def check_stable(self):
        worst = (0.0, None)  # (dist, box_id)
        for box_id in self.boxes_on_pallet_id:
            cur = self.get_box_pose(box_id)[:3]
            tgt = self.boxes_on_pallet_target_pose[box_id][:3]
            d = float(np.linalg.norm(cur - tgt))
            if d > worst[0]:
                worst = (d, int(box_id))
            if d > self.stable_thres:
                return False, worst
        return True, worst

    def sim_forward_until_unstable(
        self,
        steps: int,
        placed_box_id: int,
        placed_target_pos: np.ndarray,
        check_every: int = 1,
    ) -> bool:
        # STRICT: match env.py early-stop behavior
        check_every = max(1, int(check_every))
        for k in range(steps):
            self.sim.forward()
            self.sim.step()
            if (k + 1) % check_every != 0:
                continue

            ok, _ = self.check_stable()
            if not ok:
                return False

            cur_pos = self.get_box_pose(placed_box_id)[:3]
            d = float(np.linalg.norm(cur_pos - placed_target_pos))
            if d > self.stable_thres:
                print(f"[Unstable] placed_box drift d={d:.4f} th={self.stable_thres:.4f} cur={cur_pos} tgt={placed_target_pos}")
                return False

        return True

    def get_target_position_strict(
        self,
        box_size_after_rotate: np.ndarray,
        x: int,
        y: int,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        X = int(self.pallet_size_discrete[0])
        Y = int(self.pallet_size_discrete[1])
        H = int(self.max_pallet_height)

        x = int(np.clip(int(x), 0, X - 1))
        y = int(np.clip(int(y), 0, Y - 1))

        dx = int(box_size_after_rotate[0])
        dy = int(box_size_after_rotate[1])
        dz = int(box_size_after_rotate[2])

        if (x + dx > X) or (y + dy > Y):
            raise RuntimeError(
                f"[OutOfBounds] Box would exceed pallet boundary: "
                f"(x,y)=({x},{y}), (dx,dy)=({dx},{dy}), pallet=(X,Y)=({X},{Y})."
            )

        x2 = x + dx
        y2 = y + dy

        pallet = self.obs["pallet_obs_density"]
        place_area = pallet[x:x2, y:y2, :]
        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
        z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0

        if (z + dz > H):
            raise HeightLimitExceeded(
                f"[OutOfBoundsZ] Box would exceed max height: (x,y)=({x},{y}), (z,dz)=({z},{dz}), H={H}.",
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
    # Candidate pools (front/backlog/rehandle)
    # ---------------------------

    def _init_candidate_pools(self, rng: np.random.Generator):
        self.backlog_queue = copy.copy(self.boxes_ids)
        rng.shuffle(self.backlog_queue)

        self.front_ids = [-1 for _ in range(self.N_visible_boxes)]
        self.rehandle_pool: List[int] = []
        self.rehandle_cooldown: Dict[int, int] = {}  # box_id -> earliest step it can re-enter front

        self.step_count = 0
        self._refill_front()

    def _refill_front(self):
        for i in range(self.N_visible_boxes):
            if int(self.front_ids[i]) >= 0:
                continue

            chosen = None

            if len(self.rehandle_pool) > 0:
                for k, bid in enumerate(self.rehandle_pool):
                    ready_t = int(self.rehandle_cooldown.get(int(bid), 0))
                    if self.step_count >= ready_t:
                        chosen = int(bid)
                        self.rehandle_pool.pop(k)
                        break

            if chosen is None and len(self.backlog_queue) > 0:
                chosen = int(self.backlog_queue.pop(0))

            if chosen is None:
                break

            self.front_ids[i] = int(chosen)

    # ---------------------------
    # Observations
    # ---------------------------

    def update_obs_buffer(self):
        buf = np.zeros(self.N_visible_boxes * self.n_properties, dtype=np.float32)
        buf_phy = np.zeros(self.N_visible_boxes * self.n_physics_properties, dtype=np.float32)

        for i in range(self.N_visible_boxes):
            bid = int(self.front_ids[i])
            if bid < 0:
                continue

            props = self.id_to_properties[bid]
            buf[self.n_properties * i : self.n_properties * (i + 1)] = props

            if self.expose_physics_obs:
                softness = float(self.id_to_softness.get(bid, 0.0))
                mu = float(self.id_to_friction_mu.get(bid, 0.0))
                base = self.n_physics_properties * i
                buf_phy[base + 0] = softness
                buf_phy[base + 1] = mu

        self.obs["buffer"] = buf
        self.obs["buffer_physics"] = buf_phy

        self.obs["front_ids"] = np.array(self.front_ids, dtype=np.int32)

        self.obs["pallet_count"] = int(len(self.boxes_on_pallet_id))
        self.obs["pallet_footprints"] = list(self.pallet_footprints)

        # ✅ NEW: expose pallet_ids aligned with pallet_footprints
        self.obs["pallet_ids"] = list(self.boxes_on_pallet_id)

        # ✅ NEW: expose removable_mask aligned with pallet_footprints
        try:
            self.obs["removable_mask"] = self.remove_checker.compute_removable_mask(
                self.obs["pallet_obs_density"],
                self.pallet_footprints,
            )
        except Exception:
            self.obs["removable_mask"] = np.zeros((len(self.pallet_footprints),), dtype=np.int8)

    def save_frame(self):
        if not self.save_video:
            return
        self._update_observables()
        img = self.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        self.writer.append_data(img)

    # ---------------------------
    # Reinit / Reset
    # ---------------------------

    def _reset_boxes_to_init_pose_if_available(self):
        if not hasattr(self, "box_init_pose"):
            return
        for i in range(self.total_box_number):
            box_id = self.boxes_ids[i]
            init_pose = np.array(self.box_init_pose[i], dtype=np.float32)
            box_obj = self.id_to_box_obj[box_id]
            self.place_box(box_obj, init_pose[:3], init_pose[3:])

    def reinit(self, rng: np.random.Generator) -> Dict:
        self._reset_boxes_to_init_pose_if_available()

        X, Y = int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1])
        H = int(self.max_pallet_height)

        self.obs = {
            "pallet_obs_density": np.zeros((X, Y, H), dtype=np.float32),
            "pallet_obs_softness": np.zeros((X, Y, H), dtype=np.float32),
            "buffer": np.zeros((self.N_visible_boxes * self.n_properties,), dtype=np.float32),
            "buffer_physics": np.zeros((self.N_visible_boxes * self.n_physics_properties,), dtype=np.float32),

            "front_ids": np.full((self.N_visible_boxes,), -1, dtype=np.int32),
            "pallet_count": 0,
            "pallet_footprints": [],

            # ✅ NEW: always-present fields for heuristics/LLM
            "pallet_ids": [],
            "removable_mask": np.zeros((0,), dtype=np.int8),
        }

        self.boxes_on_pallet_id: List[int] = []
        self.boxes_on_pallet_target_pose: Dict[int, np.ndarray] = {}

        self.pallet_footprints: List[Tuple[int, int, int, int, int, int]] = []
        self.boxid_to_footprint: Dict[int, Tuple[int, int, int, int, int, int]] = {}

        self._init_candidate_pools(rng)

        self.update_obs_buffer()
        return self.obs

    # ---------------------------
    # Unpacking feasibility (top-removable)
    # ---------------------------

    def is_removable(self, pallet_index: int) -> bool:
        if len(self.boxes_on_pallet_id) == 0:
            return False
        pallet_index = int(np.clip(int(pallet_index), 0, len(self.boxes_on_pallet_id) - 1))
        box_id = int(self.boxes_on_pallet_id[pallet_index])
        fp = self.boxid_to_footprint.get(box_id, None)
        if fp is None:
            return False

        x, y, z, dx, dy, dz = fp
        H = int(self.max_pallet_height)
        top = int(z + dz)
        if top >= H:
            return True

        above = self.obs["pallet_obs_density"][x:x+dx, y:y+dy, top:H]
        return (not np.any(above > 0))

    def _find_last_removable_index(self) -> Optional[int]:
        for i in range(len(self.boxes_on_pallet_id) - 1, -1, -1):
            if self.is_removable(i):
                return int(i)
        return None

    # ---------------------------
    # REMOVE execution
    # ---------------------------

    def _teleport_to_rehandle_staging(self, box_id: int):
        """
        Teleport removed box to the OTHER SIDE of the pallet (safe staging).
        - Place on table surface (not in mid-air)
        - Arrange in a grid with enough spacing to avoid contacts/explosions
        """
        box_obj = self.id_to_box_obj[box_id]

        # How many boxes already staged (after we append, but we compute before append)
        k = len(self.rehandle_pool)  # 0-based
        cols = 6
        col = k % cols
        row = k // cols

        # --- reference: pallet pose / size ---
        pallet_c = np.array(self.pallet_position, dtype=np.float32)      # center of pallet
        pallet_half = np.array(self.pallet_size, dtype=np.float32) / 2.0 # half-extent

        # Put staging on the "other side" of pallet along +Y (you can flip sign if you want)
        # Add a margin so it's clearly outside pallet boundary.
        margin_y = 0.25  # meters, tune if needed

        # --- spacing: based on box footprint ---
        # box_obj.size is HALF-size in meters. Full size:
        full = np.array(box_obj.size, dtype=np.float32) * 2.0
        # Use max footprint to space both directions; add extra gap.
        gap = 0.3
        step_x = float(max(full[0], 0.05) + gap)
        step_y = float(max(full[1], 0.05) + gap)

        # Start point: beside pallet edge on +Y side, a bit shifted in X so grid stays on table
        base_x = float(pallet_c[0] - pallet_half[0] + 0.10)   # near pallet's -X edge
        base_y = float(pallet_c[1] + pallet_half[1] + margin_y)

        x = base_x + step_x * col
        y = base_y + step_y * row

        # --- Z: put on table surface ---
        # table surface height in world:
        table_z = float(self.table_offset[2])

        # robust: ensure box bottom is just above table
        # box_obj.bottom_offset is a 3D vector; subtracting it in your random init indicates it's "how much to lift"
        # So to place on table: table_z - bottom_offset_z + small epsilon
        eps = 0.002
        z = table_z - float(box_obj.bottom_offset[2]) + eps

        pos = np.array([x, y, z], dtype=np.float32)

        # Identity rotation is fine; if your boxes sometimes have weird inertia, you can also set a stable quat.
        quat = np.array([0, 0, 0, 1], dtype=np.float32)

        self.place_box(box_obj, pos, quat)

        # Optional: kill residual velocities hard (extra safety)
        try:
            self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))
        except Exception:
            pass


    def remove_box_from_pallet(self, pallet_index: int) -> Tuple[bool, Dict]:
        info: Dict = {}

        if len(self.boxes_on_pallet_id) == 0:
            return False, {"error": "empty_pallet"}

        pallet_index = int(np.clip(int(pallet_index), 0, len(self.boxes_on_pallet_id) - 1))

        if not self.is_removable(pallet_index):
            found = self._find_last_removable_index()
            if found is None:
                return False, {"error": "no_removable_box_all_covered"}
            pallet_index = int(found)

        box_id = int(self.boxes_on_pallet_id[pallet_index])
        fp = self.boxid_to_footprint.get(box_id, None)
        if fp is None:
            return False, {"error": "missing_footprint"}

        x, y, z, dx, dy, dz = fp

        self.obs["pallet_obs_density"][x:x+dx, y:y+dy, z:z+dz] = 0.0
        if self.expose_physics_obs:
            self.obs["pallet_obs_softness"][x:x+dx, y:y+dy, z:z+dz] = 0.0

        self.boxes_on_pallet_id.pop(pallet_index)
        self.pallet_footprints.pop(pallet_index)
        self.boxes_on_pallet_target_pose.pop(box_id, None)
        self.boxid_to_footprint.pop(box_id, None)

        self._teleport_to_rehandle_staging(box_id)
        self.rehandle_pool.append(int(box_id))
        self.rehandle_cooldown[int(box_id)] = int(self.step_count + self.rehandle_cooldown_steps)

        info["removed_box_id"] = int(box_id)
        info["removed_index"] = int(pallet_index)

        return True, info

    # ---------------------------
    # PLACE commit bookkeeping
    # ---------------------------

    def _commit_place(
        self,
        box_id: int,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        x: int, y: int, z: int,
        dx: int, dy: int, dz: int,
        box_density: float,
        box_softness: float,
    ):
        self.boxes_on_pallet_id.append(int(box_id))
        self.boxes_on_pallet_target_pose[int(box_id)] = np.concatenate([target_pos, target_quat])

        fp = (int(x), int(y), int(z), int(dx), int(dy), int(dz))
        self.pallet_footprints.append(fp)
        self.boxid_to_footprint[int(box_id)] = fp

        self.obs["pallet_obs_density"][x:x+dx, y:y+dy, z:z+dz] = float(box_density)
        if self.expose_physics_obs:
            self.obs["pallet_obs_softness"][x:x+dx, y:y+dy, z:z+dz] = float(box_softness)

    # ---------------------------
    # Reward
    # ---------------------------

    def reward_func(self, termination_reason: str) -> Tuple[float, Dict]:
        """
        termination_reason: natural language string
          - "none"
          - "unstable"
          - "success"
          - "height_oob"
        """
        code_map = {
            "none": 0,
            "unstable": 2,
            "success": 3,
            "height_oob": 5,
        }

        info: Dict[str, float] = {
            "termination_reason": str(termination_reason),
            "termination_reason_code": int(code_map.get(str(termination_reason), 0)),
            "physics_mode": str(self.physics_mode),
            "expose_physics_obs": int(self.expose_physics_obs),
        }

        if str(termination_reason) in ("unstable", "success", "height_oob"):
            metrics = compute_utilization_volume(self.obs["pallet_obs_density"])
            r = float(metrics["util"])
            info.update(metrics)
        else:
            r = 0.0

        info["reward_util_final"] = float(r)
        return float(r), info

    # ---------------------------
    # Step
    # ---------------------------

    def step(self, action: np.ndarray):
        """
        action: int32[5]
        - PLACE:  [0, slot, rot_id, x, y]
        - REMOVE: [1, pallet_index, 0, 0, 0]
        """
        self.step_count += 1

        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int32)
        action = action.astype(np.int32).reshape(-1)
        if action.size < 5:
            a2 = np.zeros(5, dtype=np.int32)
            a2[:action.size] = action
            action = a2

        op = int(action[0])

        # Always attach util_current for logging (even if truncated by max_steps)
        def _attach_util_current(info: dict) -> dict:
            try:
                metrics_now = compute_utilization_volume(self.obs["pallet_obs_density"])
                info["util_current"] = float(metrics_now["util"])
                info["pallet_count"] = int(len(self.boxes_on_pallet_id))
            except Exception:
                info["util_current"] = float(0.0)
                info["pallet_count"] = int(len(getattr(self, "boxes_on_pallet_id", [])))
            return info

        # -----------------------
        # REMOVE
        # -----------------------
        if op == 1:
            idx = int(action[1])
            ok, rinfo = self.remove_box_from_pallet(idx)
            self._settle(self.settle_steps_after_remove)

            self._refill_front()
            self.update_obs_buffer()
            self.save_frame()
            self._refill_front()
            self.update_obs_buffer()
            self.save_frame()

            reward, info = self.reward_func("none")
            info["op"] = "REMOVE"
            info["remove_ok"] = int(ok)
            if isinstance(rinfo, dict):
                info.update(rinfo)

            info = _attach_util_current(info)

            print(f"[Step] op=REMOVE idx={idx} ok={int(ok)}")
            done = False
            return self.obs, reward, done, info

        # -----------------------
        # PLACE
        # -----------------------
        slot = int(action[1])
        rot_id = int(action[2])
        x = int(action[3])
        y = int(action[4])

        slot = int(np.clip(slot, 0, self.N_visible_boxes - 1))
        rot_id = int(np.clip(rot_id, 0, 5))

        box_id = int(self.front_ids[slot])
        if box_id < 0:
            reward, info = self.reward_func("none")
            info["op"] = "PLACE"
            info["error"] = "empty_front_slot"
            self.save_frame()
            print(f"[Step] op=PLACE slot={slot} rot_id={rot_id} -> empty slot")
            done = False
            info = _attach_util_current(info)
            return self.obs, reward, done, info

        box_obj = self.id_to_box_obj[box_id]

        box_size = (np.array(box_obj.size, dtype=np.float32) * 2.0 / self.bin_size).astype(int)
        box_density = float(self.id_to_properties[box_id][3])
        box_softness = float(self.id_to_softness.get(box_id, 0.0))

        order = ORDERS[rot_id]
        target_quat = quat_xyzw_from_order(rot_id)
        size_after_rotate = box_size[order]
        dx, dy, dz = int(size_after_rotate[0]), int(size_after_rotate[1]), int(size_after_rotate[2])

        try:
            target_pos, (x2, y2, z2) = self.get_target_position_strict(size_after_rotate, x, y)
        except HeightLimitExceeded as e:
            reward, info = self.reward_func("height_oob")
            done = True
            self.save_frame()
            info["op"] = "PLACE"
            info["error"] = str(e)
            info["chosen_discrete"] = {"slot": slot, "rot_id": rot_id, "x": int(e.x), "y": int(e.y), "z": int(e.z)}
            info["box_density"] = float(box_density)
            info["box_softness"] = float(box_softness)

            info = _attach_util_current(info)

            print(
                f"[EpisodeEnd] reason=height_oob util={float(info.get('util', 0.0)):.4f} "
                f"n_boxes_on_pallet={len(self.boxes_on_pallet_id)}"
            )
            return self.obs, reward, done, info

        print(f"[Step] op=PLACE slot={slot} rot_id={rot_id} place=(x={int(x2)}, y={int(y2)}, z={int(z2)})")

        self.place_box(box_obj, target_pos, target_quat)

        # physics rollout (STRICT)
        if self.early_stop_on_unstable:
            is_stable = self.sim_forward_until_unstable(
                steps=40,
                placed_box_id=box_id,
                placed_target_pos=target_pos,
                check_every=self.early_stop_check_every,
            )
        else:
            for _ in range(40):
                self.sim.forward()
                self.sim.step()
            cur_pos = self.get_box_pose(box_id)[:3]
            is_stable = self.check_stable() and (np.linalg.norm(cur_pos - target_pos) < self.stable_thres)

        if not is_stable:
            reward, info = self.reward_func("unstable")
            done = True
            self.save_frame()
            info["op"] = "PLACE"
            info["chosen_discrete"] = {"slot": slot, "rot_id": rot_id, "x": int(x2), "y": int(y2), "z": int(z2)}
            info["box_density"] = float(box_density)
            info["box_softness"] = float(box_softness)

            info = _attach_util_current(info)

            print(
                f"[EpisodeEnd] reason=unstable util={float(info.get('util', 0.0)):.4f} "
                f"n_boxes_on_pallet={len(self.boxes_on_pallet_id)}"
            )
            return self.obs, reward, done, info

        # commit place + update front/backlog/rehandle
        self._commit_place(
            box_id=box_id,
            target_pos=target_pos,
            target_quat=target_quat,
            x=int(x2), y=int(y2), z=int(z2),
            dx=int(dx), dy=int(dy), dz=int(dz),
            box_density=float(box_density),
            box_softness=float(box_softness),
        )
        self._settle(self.settle_steps_after_place)
        self.front_ids[slot] = -1
        self._refill_front()

        self.update_obs_buffer()
        self.save_frame()

        all_front_empty = all(int(v) < 0 for v in self.front_ids)
        done = (len(self.backlog_queue) == 0) and (len(self.rehandle_pool) == 0) and all_front_empty

        term_reason = "success" if done else "none"
        reward, info = self.reward_func(term_reason)
        info["op"] = "PLACE"
        info["chosen_discrete"] = {"slot": int(slot), "rot_id": int(rot_id), "x": int(x2), "y": int(y2), "z": int(z2)}
        info["box_density"] = float(box_density)
        info["box_softness"] = float(box_softness)

        info = _attach_util_current(info)

        if term_reason == "success":
            print(
                f"[EpisodeEnd] reason=success util={float(info.get('util', 0.0)):.4f} "
                f"n_boxes_on_pallet={len(self.boxes_on_pallet_id)}"
            )

        return self.obs, reward, done, info

    # ---------------------------
    # Close (avoid EGL destructor warnings)
    # ---------------------------

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass


# ---------------------------
# Gymnasium Wrapper
# ---------------------------

class MixedBoxPlanningEnvWrapper(gym.Env):
    """
    Gym wrapper:
      - action_space: int32[5] but gymnasium Box uses float; we'll still accept int arrays.
      - step returns (obs, reward, done, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        save_video_path: Optional[str] = None,
        physics_mode: Optional[str] = None,
        expose_physics_obs: bool = True,
        rehandle_cooldown_steps: int = 5,

        
    ):
        super().__init__()
        self.env = MixedBoxPlanning(
            save_video_path=save_video_path,
            init_box_pose_path=None,
            physics_mode=physics_mode,
            expose_physics_obs=expose_physics_obs,
            rehandle_cooldown_steps=rehandle_cooldown_steps,
        )

        X, Y = int(self.env.pallet_size_discrete[0]), int(self.env.pallet_size_discrete[1])
        H = int(self.env.max_pallet_height)
        N = int(self.env.N_visible_boxes)

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([1, max(N - 1, 0), 5, max(X - 1, 0), max(Y - 1, 0)], dtype=np.int32),
            dtype=np.int32,
        )

        obs_spaces = {
            "pallet_obs_density": spaces.Box(low=0.0, high=10.0, shape=(X, Y, H), dtype=np.float32),
            "pallet_obs_softness": spaces.Box(low=0.0, high=10.0, shape=(X, Y, H), dtype=np.float32),
            "buffer": spaces.Box(low=0.0, high=10.0, shape=(N * self.env.n_properties,), dtype=np.float32),
            "buffer_physics": spaces.Box(low=0.0, high=10.0, shape=(N * self.env.n_physics_properties,), dtype=np.float32),

            "front_ids": spaces.Box(low=-1, high=10**9, shape=(N,), dtype=np.int32),
            "pallet_count": spaces.Discrete(10**9),
        }
        self.observation_space = spaces.Dict(obs_spaces)
        
    def _settle(self, steps: int = 10):
        steps = int(max(0, steps))
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reinit(rng=self.np_random)
        return obs, {}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass
