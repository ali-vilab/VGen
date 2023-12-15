import torch
import math

from utils.registry_class import DIFFUSION
from .schedules import beta_schedule
from .losses import kl_divergence, discretized_gaussian_log_likelihood
# from .dpm_solver import NoiseScheduleVP, model_wrapper_guided_diffusion, model_wrapper, DPM_Solver

# def _i(tensor, t, x):
#     r"""Index tensor using t and format the output according to x.
#     """
#     shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
#     return tensor[t].view(shape).to(x)

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


@DIFFUSION.register_class()
class DiffusionDDIM(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 **kwargs):
        # check input
        # check input
        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # eps
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength # 0.0

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise


    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) # guide_scale=9.0
        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)
        
        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'v':
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    
    @torch.no_grad()
    def ddim_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
        return xt
    
    @torch.no_grad()
    def ddim_reverse_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            torch.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)
        
        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt
    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def loss(self, x0, t, model, model_kwargs={}, noise=None, weight = None, use_div_loss= False):

        # noise = torch.randn_like(x0) if noise is None else noise # [80, 4, 8, 32, 32]
        noise = self.sample_loss(x0, noise)

        xt = self.q_sample(x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model, model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1', 'rescaled_l1']: # self.loss_type: mse
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']: # self.var_type: 'fixed_small'
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            # target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            target = {
                'eps': noise, 
                'x0': x0, 
                'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0], 
                'v':_i(self.sqrt_alphas_cumprod, t, xt) * noise - _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * x0}[self.mean_type]
            loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2).abs().flatten(1).mean(dim=1)
            if weight is not None:
                loss = loss*weight   

            # div loss
            if use_div_loss and self.mean_type == 'eps' and x0.shape[2]>1:
                 
                # derive  x0
                x0_ = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out

                # # derive xt_1, set eta=0 as ddim
                # alphas_prev = _i(self.alphas_cumprod, (t - 1).clamp(0), xt)
                # direction = torch.sqrt(1 - alphas_prev) * out
                # xt_1 = torch.sqrt(alphas_prev) * x0_ + direction

                # ncfhw, std on f
                div_loss = 0.001/(x0_.std(dim=2).flatten(1).mean(dim=1)+1e-4)
                # print(div_loss,loss)
                loss = loss+div_loss

            # total loss
            loss = loss + loss_vlb
        elif self.loss_type in ['charbonnier']:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            loss = torch.sqrt((out - target)**2 + self.epsilon)
            if weight is not None:
                loss = loss*weight
            loss = loss.flatten(1).mean(dim=1)
            
            # total loss
            loss = loss + loss_vlb
        return loss

    def variational_lower_bound(self, x0, xt, t, model, model_kwargs={}, clamp=None, percentile=None):
        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2)
        kl = kl.flatten(1).mean(dim=1) / math.log(2.0)
        
        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(x0, mean=mu2, log_scale=0.5 * log_var2)
        nll = nll.flatten(1).mean(dim=1) / math.log(2.0)

        # NLL for p(x0 | x1) and KL otherwise
        vlb = torch.where(t == 0, nll, kl)
        return vlb, x0
    
    @torch.no_grad()
    def variational_lower_bound_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None):
        r"""Compute the entire variational lower bound, measured in bits-per-dim.
        """
        # prepare input and output
        b = x0.size(0)
        metrics = {'vlb': [], 'mse': [], 'x0_mse': []}

        # loop
        for step in torch.arange(self.num_timesteps).flip(0):
            # compute VLB
            t = torch.full((b, ), step, dtype=torch.long, device=x0.device)
            # noise = torch.randn_like(x0)
            noise = self.sample_loss(x0)
            xt = self.q_sample(x0, t, noise)
            vlb, pred_x0 = self.variational_lower_bound(x0, xt, t, model, model_kwargs, clamp, percentile)

            # predict eps from x0
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)

            # collect metrics
            metrics['vlb'].append(vlb)
            metrics['x0_mse'].append((pred_x0 - x0).square().flatten(1).mean(dim=1))
            metrics['mse'].append((eps - noise).square().flatten(1).mean(dim=1))
        metrics = {k: torch.stack(v, dim=1) for k, v in metrics.items()}

        # compute the prior KL term for VLB, measured in bits-per-dim
        mu, _, log_var = self.q_mean_variance(x0, t)
        kl_prior = kl_divergence(mu, log_var, torch.zeros_like(mu), torch.zeros_like(log_var))
        kl_prior = kl_prior.flatten(1).mean(dim=1) / math.log(2.0)

        # update metrics
        metrics['prior_bits_per_dim'] = kl_prior
        metrics['total_bits_per_dim'] = metrics['vlb'].sum(dim=1) + kl_prior
        return metrics

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()
