import torch

def to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_device(u, device, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_device(v, device, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor) and batch.device != device:
        batch = batch.to(device, non_blocking=non_blocking)
    else:
        return batch
    return batch
