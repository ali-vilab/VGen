from .registry import Registry, build_from_config

def build_func(cfg, registry, **kwargs):
    """
    Except for config, if passing a list of dataset config, then return the concat type of it
    """
    return build_from_config(cfg, registry, **kwargs)

AUTO_ENCODER = Registry("AUTO_ENCODER", build_func=build_func)
DATASETS = Registry("DATASETS", build_func=build_func)
DIFFUSION = Registry("DIFFUSION", build_func=build_func)
DISTRIBUTION = Registry("DISTRIBUTION", build_func=build_func)
EMBEDDER = Registry("EMBEDDER", build_func=build_func)
ENGINE = Registry("ENGINE", build_func=build_func)
INFER_ENGINE = Registry("INFER_ENGINE", build_func=build_func)
MODEL = Registry("MODEL", build_func=build_func)
PRETRAIN = Registry("PRETRAIN", build_func=build_func)
VISUAL = Registry("VISUAL", build_func=build_func)
