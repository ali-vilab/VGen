import math
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['AnnealingLR']

class AnnealingLR(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, decay_mode='cosine', min_lr=0.0, last_step=-1):
        assert decay_mode in ['linear', 'cosine', 'none']
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_mode = decay_mode
        self.min_lr = min_lr
        self.current_step = last_step + 1
        self.step(self.current_step)
    
    def get_lr(self):
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            ratio = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            ratio = min(1.0, max(0.0, ratio))
            if self.decay_mode == 'linear':
                return self.base_lr * (1 - ratio)
            elif self.decay_mode == 'cosine':
                return self.base_lr * (math.cos(math.pi * ratio) + 1.0) / 2.0
            else:
                return self.base_lr
    
    def step(self, current_step=None):
        if current_step is None:
            current_step = self.current_step + 1
        self.current_step = current_step
        new_lr = max(self.min_lr, self.get_lr())
        if isinstance(self.optimizer, list):
            for o in self.optimizer:
                for group in o.param_groups:
                    group['lr'] = new_lr
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
    
    def state_dict(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'decay_mode': self.decay_mode,
            'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.decay_mode = state_dict['decay_mode']
        self.current_step = state_dict['current_step']
