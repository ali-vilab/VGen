"""
GaussianDiffusion wraps operators for denoising diffusion models, including the
diffusion and denoising processes, as well as the loss evaluation.
"""
import torch
import torchsde
import random
from tqdm.auto import trange


__all__ = ['GaussianDiffusion']


def _i(tensor, t, x):
    """
    Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    """
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(
            t0, w0, t1, entropy=s, **kwargs
        ) for s in seed]
    
    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """
    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0 = self.transform(torch.as_tensor(sigma_min))
        t1 = self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)
    
    def __call__(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma ** 2 + 1. ** 2) ** 0.5
    return c_out, c_in


@torch.no_grad()
def sample_dpmpp_2m_sde(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    solver_type='midpoint',
    show_progress=True
):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {'heun', 'midpoint'}

    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + \
                    (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * \
                            (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * \
                            (1 / r) * (denoised - old_denoised)

                x = x + noise_sampler(
                    sigmas[i],
                    sigmas[i + 1]
                ) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h
    return x


class GaussianDiffusion(object):

    def __init__(self, sigmas, prediction_type='eps'):
        assert prediction_type in {'x0', 'eps', 'v'}
        self.sigmas = sigmas.float()                        # noise coefficients
        self.alphas = torch.sqrt(1 - sigmas ** 2).float()   # signal coefficients
        self.num_timesteps = len(sigmas)
        self.prediction_type = prediction_type

    def diffuse(self, x0, t, noise=None):
        """
        Add Gaussian noise to signal x0 according to:
        q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt
    
    def denoise(
        self,
        xt,
        t,
        s,
        model,
        model_kwargs={},
        guide_scale=None,
        guide_rescale=None,
        clamp=None,
        percentile=None
    ):
        """
        Apply one step of denoising from the posterior distribution q(x_s | x_t, x0).
        Since x0 is not available, estimate the denoising results using the learned
        distribution p(x_s | x_t, \hat{x}_0 == f(x_t)).
        """
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s ** 2)

        # precompute variables
        betas = 1 - (alphas / alphas_s) ** 2
        coef1 = betas * alphas_s / sigmas ** 2
        coef2 = (alphas * sigmas_s ** 2) / (alphas_s * sigmas ** 2)
        var = betas * (sigmas_s / sigmas) ** 2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t=t, **model_kwargs[0])
            if guide_scale == 1.:
                out = y_out
            else:
                u_out = model(xt, t=t, **model_kwargs[1])
                out = u_out + guide_scale * (y_out - u_out)

                # rescale the output according to arXiv:2305.08891
                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (y_out.flatten(1).std(dim=1) / (
                        out.flatten(1).std(dim=1) + 1e-12
                    )).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        
        # compute x0
        if self.prediction_type == 'x0':
            x0 = out
        elif self.prediction_type == 'eps':
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == 'v':
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(
                f'prediction_type {self.prediction_type} not implemented'
            )
        
        # restrict the range of x0
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        
        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    @torch.no_grad()
    def sample(
        self,
        noise,
        model,
        model_kwargs={},
        condition_fn=None,
        guide_scale=None,
        guide_rescale=None,
        clamp=None,
        percentile=None,
        solver='euler_a',
        steps=20,
        t_max=None,
        t_min=None,
        discretization=None,
        discard_penultimate_step=None,
        return_intermediate=None,
        show_progress=False,
        seed=-1,
        **kwargs
    ):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')

        # function of diffusion solver
        solver_fn = {
            # 'heun': sample_heun,
            'dpmpp_2m_sde': sample_dpmpp_2m_sde
        }[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2 ** 31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2',
                'dpm2_ancestral',
                'dpmpp_2m_sde',
                'dpm2_karras',
                'dpm2_ancestral_karras',
                'dpmpp_2m_sde_karras'
            ) else False
        
        # function for denoising xt to get x0
        intermediates = []
        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            x0 = self.denoise(
                xt, t, None, model, model_kwargs, guide_scale, guide_rescale, clamp,
                percentile
            )[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            return x0
        
        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(
                    t_min, t_max + 1, (t_max - t_min + 1) / steps
                ).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1, -((t_max - t_min + 1) / steps))
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented'
                )
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps, dtype=torch.float32, device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.
                ).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas, sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.
                ).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        
        # sampling
        x0 = solver_fn(
            noise,
            model_fn,
            sigmas,
            show_progress=show_progress,
            **kwargs
        )
        return (x0, intermediates) if return_intermediate is not None else x0
    
    @torch.no_grad()
    def ddim_reverse_sample(
        self,
        xt,
        t,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        guide_scale=None,
        guide_rescale=None,
        ddim_timesteps=20,
        reverse_steps=600
        ):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = reverse_steps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        # _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)
        _, _, _, x0, eps = self.denoise(
                xt, t, None, model, model_kwargs, guide_scale, guide_rescale, clamp,
                percentile
            )
        # derive variables
        s = (t + stride).clamp(0, reverse_steps-1)
        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s ** 2)
        
        # reverse sample
        mu = alphas_s * x0 + sigmas_s * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(
        self,
        x0,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        guide_scale=None,
        guide_rescale=None,
        ddim_timesteps=20,
        reverse_steps=600
        ):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, reverse_steps, reverse_steps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, guide_rescale, ddim_timesteps, reverse_steps)
        return xt
    
    def _sigma_to_t(self, sigma):
        if sigma == float('inf'):
            t = torch.full_like(sigma, len(self.sigmas) - 1)
        else:
            log_sigmas = torch.sqrt(
                self.sigmas ** 2 / (1 - self.sigmas ** 2)
            ).log().to(sigma)
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(
                max=log_sigmas.shape[0] - 2
            )
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sigmas ** 2 / (1 - self.sigmas ** 2)).log().to(t)
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma) | torch.isinf(log_sigma)] = float('inf')
        return log_sigma.exp()
    
    def prev_step(self, model_out, t, xt, inference_steps=50):
        prev_t = t - self.num_timesteps // inference_steps

        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_prev = _i(self.alphas, prev_t.clamp(0), xt)
        alphas_prev[prev_t < 0] = 1.
        sigmas_prev = torch.sqrt(1 - alphas_prev ** 2)
        
        x0 = alphas * xt - sigmas * model_out
        eps = (xt - alphas * x0) / sigmas
        prev_sample = alphas_prev * x0 + sigmas_prev * eps
        return prev_sample
    
    def next_step(self, model_out, t, xt, inference_steps=50):
        t, next_t = min(t - self.num_timesteps // inference_steps, 999), t

        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_next = _i(self.alphas, next_t.clamp(0), xt)
        alphas_next[next_t < 0] = 1.
        sigmas_next = torch.sqrt(1 - alphas_next ** 2)
        
        x0 = alphas * xt - sigmas * model_out
        eps = (xt - alphas * x0) / sigmas
        next_sample = alphas_next * x0 + sigmas_next * eps
        return next_sample
    
    def get_noise_pred_single(self, xt, t, model, model_kwargs):
        assert isinstance(model_kwargs, dict)
        out = model(xt, t=t, **model_kwargs)
        return out
    
    
