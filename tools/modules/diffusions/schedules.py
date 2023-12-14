import math
import torch


def beta_schedule(schedule='cosine',
                   num_timesteps=1000,
                   zero_terminal_snr=False,
                   **kwargs):
    # compute betas
    betas = {
        'logsnr_cosine_interp': logsnr_cosine_interp_schedule,
        'linear': linear_schedule,
        'linear_sd': linear_sd_schedule,
        'quadratic': quadratic_schedule,
        'cosine': cosine_schedule
    }[schedule](num_timesteps, **kwargs)

    if zero_terminal_snr and betas.max() != 1.0:
        betas = rescale_zero_terminal_snr(betas)

    return betas


def linear_schedule(num_timesteps, init_beta, last_beta,  **kwargs):
    scale = 1000.0 / num_timesteps
    init_beta = init_beta or scale * 0.0001
    ast_beta = last_beta or scale * 0.02
    return torch.linspace(init_beta, last_beta, num_timesteps, dtype=torch.float64)

def logsnr_cosine_interp_schedule(
        num_timesteps,
        scale_min=2,
        scale_max=4,
        logsnr_min=-15,
        logsnr_max=15,
        **kwargs):
    return logsnrs_to_sigmas(
        _logsnr_cosine_interp(num_timesteps, logsnr_min, logsnr_max, scale_min, scale_max))

def linear_sd_schedule(num_timesteps, init_beta, last_beta,  **kwargs):
    return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2


def quadratic_schedule(num_timesteps, init_beta, last_beta,  **kwargs):
    init_beta = init_beta or 0.0015
    last_beta = last_beta or 0.0195
    return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2


def cosine_schedule(num_timesteps, cosine_s=0.008, **kwargs):
    betas = []
    for step in range(num_timesteps):
        t1 = step / num_timesteps
        t2 = (step + 1) / num_timesteps
        fn = lambda u: math.cos((u + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
    return torch.tensor(betas, dtype=torch.float64)


# def cosine_schedule(n, cosine_s=0.008, **kwargs):
#     ramp = torch.linspace(0, 1, n + 1)
#     square_alphas = torch.cos((ramp + cosine_s) / (1 + cosine_s) * torch.pi / 2) ** 2
#     betas = (1 - square_alphas[1:] / square_alphas[:-1]).clamp(max=0.999)
#     return betas_to_sigmas(betas)


def betas_to_sigmas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))


def sigmas_to_betas(sigmas):
    square_alphas = 1 - sigmas**2
    betas = 1 - torch.cat(
        [square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
    return betas



def sigmas_to_logsnrs(sigmas):
    square_sigmas = sigmas**2
    return torch.log(square_sigmas / (1 - square_sigmas))


def _logsnr_cosine(n, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_min))
    t_max = math.atan(math.exp(-0.5 * logsnr_max))
    t = torch.linspace(1, 0, n)
    logsnrs = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    return logsnrs


def _logsnr_cosine_shifted(n, logsnr_min=-15, logsnr_max=15, scale=2):
    logsnrs = _logsnr_cosine(n, logsnr_min, logsnr_max)
    logsnrs += 2 * math.log(1 / scale)
    return logsnrs

def karras_schedule(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ramp = torch.linspace(1, 0, n)
    min_inv_rho = sigma_min**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    sigmas = torch.sqrt(sigmas**2 / (1 + sigmas**2))
    return sigmas

def _logsnr_cosine_interp(n,
                          logsnr_min=-15,
                          logsnr_max=15,
                          scale_min=2,
                          scale_max=4):
    t = torch.linspace(1, 0, n)
    logsnrs_min = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_min)
    logsnrs_max = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_max)
    logsnrs = t * logsnrs_min + (1 - t) * logsnrs_max
    return logsnrs


def logsnrs_to_sigmas(logsnrs):
    return torch.sqrt(torch.sigmoid(-logsnrs))


def rescale_zero_terminal_snr(betas):
    """
    Rescale Schedule to Zero Terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values. 8 alphas_bar_sqrt_0 = a
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

