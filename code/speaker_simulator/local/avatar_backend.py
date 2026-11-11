import ast
import json
from types import SimpleNamespace

import torch
from networks.dual_branch import DualBranchUNet
from scene import FlameGaussianModel



def load_checkpoint(model):
    from utils.system_utils import searchForMaxIteration
    tree = ast.parse((model/'cfg_args').read_text(), mode='eval').body
    args = SimpleNamespace(**{node.arg: ast.literal_eval(node.value) for node in tree.keywords})
    iteration = searchForMaxIteration(str(model/'point_cloud'))
    gaussians = FlameGaussianModel(args.sh_degree, args.disable_flame_static_offset)
    gaussians.load_ply(str(model/f'point_cloud/iteration_{iteration}/point_cloud.ply'), has_target=False)
    saved = model/f'param/iteration_{iteration}'
    state = saved/'render_state.json'
    gaussians.active_sh_degree = json.loads(state.read_text())['active_sh_degree'] if state.exists() else min(iteration//1000, args.sh_degree)
    uv = torch.load(saved/'uv_coords.pt', map_location='cuda')
    network = DualBranchUNet(device='cuda', uv_mask=gaussians.get_uv_mask(), uv_sample_coords=uv)
    network.load_state_dict(torch.load(saved/'dual_branch.pth', map_location='cuda'))
    network.cuda().eval(); network.forward_once()
    gaussians.select_mesh_by_timestep(0)
    gaussians.get_position_map()
    gaussians.map_generator.position_map = network.position_map
    return gaussians, network, args, iteration
