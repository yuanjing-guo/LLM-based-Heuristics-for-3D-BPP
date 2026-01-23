import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur: BaseHeuristic, obs: dict):
    box_slot, rot_id, x, y = -1, 0, 0, 0
    pallet = obs['pallet_obs_density']
    buffer = obs['buffer']

    if buffer.size == 0:
        return (box_slot, rot_id, x, y)

    for slot_i in range(len(buffer)):
        if heur.slot_is_empty(obs, slot_i):
            continue
        props = heur.get_slot_props(obs, slot_i)
        size_bins = heur.props_to_size_bins(props)
        if size_bins.size == 0:
            continue
        if not np.all(size_bins > 0):
            continue

        for rot in range(6):
            dx, dy, dz = heur.rotate_size_bins(size_bins, rot)
            for xi in range(heur.X - dx + 1):
                for yi in range(heur.Y - dy + 1):
                    if not heur.feasibility.is_within_pallet(xi, yi, dx, dy):
                        continue
                    if heur.feasibility.is_feasible(pallet, xi, yi, dx, dy, dz, 0):
                        return (slot_i, rot, xi, yi)

    return (box_slot, rot_id, x, y)

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
