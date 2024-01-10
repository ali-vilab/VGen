import yaml
from copy import deepcopy, copy


# def get prior and ldm config
def assign_prior_mudule_cfg(cfg):
    '''
    '''
    # 
    prior_cfg = deepcopy(cfg)
    vldm_cfg = deepcopy(cfg)

    with open(cfg.prior_cfg, 'r') as f:
        _cfg_update = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # _cfg_update = _cfg_update.cfg_dict
        for k, v in _cfg_update.items():
            if isinstance(v, dict) and k in cfg:
                prior_cfg[k].update(v)
            else:
                prior_cfg[k] = v
    
    with open(cfg.vldm_cfg, 'r') as f:
        _cfg_update = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # _cfg_update = _cfg_update.cfg_dict
        for k, v in _cfg_update.items():
            if isinstance(v, dict) and k in cfg:
                vldm_cfg[k].update(v)
            else:
                vldm_cfg[k] = v

    return prior_cfg, vldm_cfg


# def get prior and ldm config
def assign_vldm_vsr_mudule_cfg(cfg):
    '''
    '''
    # 
    vldm_cfg = deepcopy(cfg)
    vsr_cfg = deepcopy(cfg)
    
    with open(cfg.vldm_cfg, 'r') as f:
        _cfg_update = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # _cfg_update = _cfg_update.cfg_dict
        for k, v in _cfg_update.items():
            if isinstance(v, dict) and k in cfg:
                vldm_cfg[k].update(v)
            else:
                vldm_cfg[k] = v
    
    with open(cfg.vsr_cfg, 'r') as f:
        _cfg_update = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # _cfg_update = _cfg_update.cfg_dict
        for k, v in _cfg_update.items():
            if isinstance(v, dict) and k in cfg:
                vsr_cfg[k].update(v)
            else:
                vsr_cfg[k] = v

    return vldm_cfg, vsr_cfg


# def get prior and ldm config
def assign_signle_cfg(cfg, _cfg_update, tname):
    '''
    '''
    # 
    vldm_cfg = deepcopy(cfg)    
    with open(_cfg_update[tname], 'r') as f:
        _cfg_update = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # _cfg_update = _cfg_update.cfg_dict
        for k, v in _cfg_update.items():
            if isinstance(v, dict) and k in cfg:
                vldm_cfg[k].update(v)
            else:
                vldm_cfg[k] = v
    return vldm_cfg