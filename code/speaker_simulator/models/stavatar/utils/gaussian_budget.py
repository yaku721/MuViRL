import torch


def limit_candidates(mask, scores, current_count, limit, increase=1):
    if not limit:
        return mask
    available = max(0, (limit-current_count)//increase)
    candidates = mask.nonzero(as_tuple=True)[0]
    if len(candidates) <= available:
        return mask
    selected = torch.zeros_like(mask)
    if available:
        order = torch.argsort(scores[candidates], descending=True, stable=True)[:available]
        selected[candidates[order]] = True
    return selected


@torch.no_grad()
def initialize_surface_gaussians(model, count):
    from torch import nn
    from utils.mesh_sampling import reweight_uvcoords_by_barycoords

    model.select_mesh_by_timestep(0)
    base_count = len(model._xyz)
    if count < base_count:
        raise ValueError(f'At least {base_count} Gaussians are required for FLAME face coverage')
    triangles = model.triangles.squeeze(0)
    area = torch.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], dim=-1).norm(dim=-1)
    extra = count-base_count
    quantiles = (torch.arange(extra, device='cuda', dtype=torch.float64)+0.5)/max(extra, 1)
    cumulative = (area.double()/area.double().sum()).cumsum(0)
    parents = torch.cat([torch.arange(base_count, device='cuda'), torch.searchsorted(cumulative, quantiles).clamp_max(base_count-1)])
    binding = model.binding[parents]
    counts = torch.bincount(binding, minlength=len(model.binding_counter))
    indices = torch.arange(count, device='cuda', dtype=torch.float64)+1
    u, v = (indices*0.7548776662466927).frac(), (indices*0.5698402909980532).frac()
    s = u.sqrt()
    bary = torch.stack([1-s, s*(1-v), s*v], dim=-1).float()
    bary[:base_count] = 1/3
    world = (triangles[binding]*bary[..., None]).sum(1)
    delta = (world-model.face_center[binding])/model.face_scaling[binding]
    local = torch.bmm(model.face_orien_mat[binding].transpose(1, 2), delta[..., None]).squeeze(-1)
    for name in ['_features_dc', '_features_rest', '_opacity', '_rotation']:
        setattr(model, name, nn.Parameter(getattr(model, name)[parents].clone()))
    model._xyz = nn.Parameter(local)
    model._scaling = nn.Parameter(model._scaling[parents].clone()-0.5*counts[binding, None].float().log())
    model.binding = binding
    model.binding_counter = counts.to(torch.int32)
    model.max_radii2D = torch.zeros(count, device='cuda')
    uv = reweight_uvcoords_by_barycoords(model.flame_model.verts_uvs.cuda(),
                                        model.flame_model.textures_idx.cuda(), binding, bary)[..., :2]
    return uv.contiguous()
