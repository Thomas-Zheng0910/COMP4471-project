"""
Simplified distributed utilities - single process stubs.
No torch.distributed, no DDP, no SLURM.
"""


def is_main_process():
    return True


def get_rank():
    return 0


def get_world_size():
    return 1


def barrier():
    pass


def sync_tensor_across_gpus(x):
    return x


def get_dist_info():
    return 0, 1


def setup_slurm(*args, **kwargs):
    pass


def setup_multi_processes(*args, **kwargs):
    pass
