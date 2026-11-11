import hashlib
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


PARAMETERS = {'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
              'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation'}
STATISTICS = ['binding', 'binding_counter', 'max_radii2D', 'xyz_gradient_accum',
              'denom', 'accum_gs_image_error', 'accum_gs_edge_error',
              'burst_indices', 'burst_errors']


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


class DistributedTraining:
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.enabled = self.world_size > 1
        self.primary = self.rank == 0
        self.topology_updates = 0
        self.frames_seen = 0
        torch.cuda.set_device(self.local_rank)
        if self.enabled:
            dist.init_process_group('nccl')

    def barrier(self):
        if self.enabled:
            dist.barrier()

    @torch.no_grad()
    def initialize(self, gaussians, network):
        if self.enabled:
            values = [p for group in gaussians.optimizer.param_groups for p in group['params']]
            values.extend(network.state_dict().values())
            for value in values:
                if value.numel():
                    contiguous = value.contiguous()
                    dist.broadcast(contiguous, 0)
                    value.copy_(contiguous)

    def loader(self, dataset, clusters, epochs, refine_epochs, workers):
        rng = np.random.default_rng(58)
        plan, indices = [], []
        groups = [(list(group), epochs - refine_epochs, True) for group in clusters]
        if refine_epochs:
            groups.append((list(range(len(dataset))), refine_epochs, False))
        for members, count, clustered in groups:
            for epoch in range(count):
                order = rng.permutation(members).tolist()
                batches = [order[i:i+self.world_size] for i in range(0, len(order), self.world_size)]
                for step, batch in enumerate(batches):
                    active = self.rank < len(batch)
                    indices.append(int(batch[self.rank] if active else batch[0]))
                    plan.append({'active': active, 'weight': self.world_size / len(batch) if active else 0.,
                                 'collect': clustered and epoch < 2,
                                 'densify': clustered and epoch == 1 and step == len(batches)-1,
                                 'frames': len(batch)})
        loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0)
        return iter(zip(loader, plan)), len(plan)

    @torch.no_grad()
    def gradients(self, gaussians, network):
        if not self.enabled:
            return
        params = [p for group in gaussians.optimizer.param_groups for p in group['params']]
        params.extend(network.parameters())
        bucket, size = [], 0
        def reduce(values):
            if not values:
                return
            flat = torch.cat([p.grad.flatten() for p in values])
            dist.all_reduce(flat)
            flat.div_(self.world_size)
            offset = 0
            for p in values:
                p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
                offset += p.numel()
        for p in params:
            if not p.requires_grad or not p.numel():
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            bucket.append(p)
            size += p.numel()
            if size >= 4 * 1024 * 1024:
                reduce(bucket)
                bucket, size = [], 0
        reduce(bucket)

    @torch.no_grad()
    def densify(self, gaussians, threshold, extent):
        for name in ['xyz_gradient_accum', 'denom', 'accum_gs_image_error', 'accum_gs_edge_error']:
            dist.all_reduce(getattr(gaussians, name))
        dist.all_reduce(gaussians.max_radii2D, op=dist.ReduceOp.MAX)
        count = torch.tensor(gaussians.iterations, device='cuda', dtype=torch.long)
        dist.all_reduce(count)
        gaussians.iterations = int(count.item())
        burst = torch.zeros(len(gaussians._xyz), device='cuda')
        burst[gaussians.burst_indices] = gaussians.burst_errors
        dist.all_reduce(burst, op=dist.ReduceOp.MAX)
        gaussians.burst_errors, gaussians.burst_indices = torch.topk(burst, min(gaussians.burst_k, len(burst)))
        # Use the same mesh scale when making the topology decision.
        gaussians.select_mesh_by_timestep(0)
        payload = [None]
        if self.primary:
            gaussians.densify_and_prune(threshold, 0.01, extent, None)
            payload[0] = cpu_tree({'parameters': {key: getattr(gaussians, name) for key, name in PARAMETERS.items()},
                                  'statistics': {name: getattr(gaussians, name) for name in STATISTICS},
                                  'iterations': gaussians.iterations,
                                  'optimizer': gaussians.optimizer.state_dict()})
        dist.broadcast_object_list(payload, src=0)
        if not self.primary:
            state = payload[0]
            for group in gaussians.optimizer.param_groups:
                if group['name'] in PARAMETERS:
                    p = torch.nn.Parameter(state['parameters'][group['name']].cuda())
                    group['params'] = [p]
                    setattr(gaussians, PARAMETERS[group['name']], p)
            gaussians.optimizer.state.clear()
            gaussians.optimizer.load_state_dict(state['optimizer'])
            for name, value in state['statistics'].items():
                setattr(gaussians, name, value.cuda() if torch.is_tensor(value) else value)
            gaussians.iterations = state['iterations']
        self.topology_updates += 1
        self.verify(gaussians)

    def verify(self, gaussians, network=None):
        digest = hashlib.sha256()
        values = [getattr(gaussians, attr) for attr in PARAMETERS.values()]
        if network is not None:
            values.extend(network.state_dict().values())
        for value in values:
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        fingerprints = [None] * self.world_size
        dist.all_gather_object(fingerprints, digest.hexdigest())
        if len(set(fingerprints)) != 1:
            raise RuntimeError('Distributed model parameters diverged across ranks')
        return fingerprints[0]

    def finish(self):
        if self.enabled:
            dist.destroy_process_group()
