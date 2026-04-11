"""
Monkey-patch hooks for Tiled Diffusion.
Hooks into ComfyUI's sampling functions to capture sigmas and
defer ControlNet evaluation for tile-based processing.

Ported from ComfyUI-TiledDiffusion (shiimizu) — SAG/Gligen patches excluded.
"""

import comfy.samplers


class Store:
    """Global state storage for inter-hook communication."""
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

store = Store()


def _delattr(obj, attr):
    try:
        if hasattr(obj, attr):
            delattr(obj, attr)
    except Exception:
        ...


# ==================== Hook: KSAMPLER.sample ====================
# Captures sigmas, model_options, extra_args into `store`
# so that SpotDiffusion can access the full sigma schedule.

def KSAMPLER_sample(*args, **kwargs):
    orig_fn = store.KSAMPLER_sample
    extra_args = None
    model_options = None
    try:
        extra_args = kwargs['extra_args'] if 'extra_args' in kwargs else args[3]
        model_options = extra_args['model_options']
    except Exception:
        ...
    if model_options is not None and 'tiled_diffusion' in model_options and extra_args is not None:
        sigmas_ = kwargs['sigmas'] if 'sigmas' in kwargs else args[2]
        sigmas_all = model_options.pop('sigmas', None)
        sigmas = sigmas_all if sigmas_all is not None else sigmas_
        store.sigmas = sigmas
        store.model_options = model_options
        store.extra_args = extra_args
    else:
        for attr in ['sigmas', 'model_options', 'extra_args']:
            _delattr(store, attr)
    return orig_fn(*args, **kwargs)


# ==================== Hook: KSampler.sample ====================
# Passes sigmas into model_options so KSAMPLER_sample can capture them.

def KSampler_sample(*args, **kwargs):
    orig_fn = store.KSampler_sample
    self = args[0]
    model_patcher = getattr(self, 'model', None)
    model_options = getattr(model_patcher, 'model_options', None)
    if model_options is not None and 'tiled_diffusion' in model_options:
        sigmas = None
        try:
            sigmas = kwargs['sigmas'] if 'sigmas' in kwargs else args[10]
        except Exception:
            ...
        if sigmas is None:
            sigmas = getattr(self, 'sigmas', None)
        if sigmas is not None:
            model_options = model_options.copy()
            model_options['sigmas'] = sigmas
            self.model.model_options = model_options
    return orig_fn(*args, **kwargs)


# ==================== Hook: get_area_and_mult ====================
# Defers ControlNet's get_control so TiledDiffusion can call it
# per-tile with cropped hint tensors.

def get_area_and_mult(*args, **kwargs):
    conds = kwargs['conds'] if 'conds' in kwargs else args[0]
    if (model_options := getattr(store, 'model_options', None)) is not None and 'tiled_diffusion' in model_options:
        if 'control' in conds:
            control = conds['control']
            if not hasattr(control, 'get_control_orig'):
                control.get_control_orig = control.get_control
            control.get_control = lambda *a, **kw: control
    else:
        if 'control' in conds:
            control = conds['control']
            if hasattr(control, 'get_control_orig') and control.get_control != control.get_control_orig:
                control.get_control = control.get_control_orig
    return store.get_area_and_mult(*args, **kwargs)


# ==================== Register hooks ====================

def register_hooks():
    patches = [
        (comfy.samplers.KSampler, 'sample', KSampler_sample),
        (comfy.samplers.KSAMPLER, 'sample', KSAMPLER_sample),
        (comfy.samplers, 'get_area_and_mult', get_area_and_mult),
    ]
    for parent, fn_name, fn_patch in patches:
        if not hasattr(parent, f"_{fn_name}"):
            setattr(store, f"_{fn_name}", getattr(parent, fn_name))
        setattr(store, fn_patch.__name__, getattr(parent, fn_name))
        setattr(parent, fn_name, fn_patch)

register_hooks()


# ==================== Patch pre_run_control ====================

def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            try:
                x['control'].cleanup()
            except Exception:
                ...
            x['control'].pre_run(model, percent_to_timestep_function)

comfy.samplers.pre_run_control = pre_run_control
