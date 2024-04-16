from dataclasses import dataclass
import random
import math
import statistics

import torch
from torch.nn.functional import conv2d, l1_loss
from tqdm import tqdm

from .data import SceneData
from .gaussians import Gaussians, RasterOutputs


gaussian_fn = lambda x, mu, sd: math.exp(-((x - mu) ** 2) / (2 * sd**2))


def ssim(img1, img2, win_size: int = 11, sigma: float = 1.5):
    C = img1.size(-3)  # channels
    # create window
    gauss = torch.tensor(
        [gaussian_fn(x, win_size // 2, sigma) for x in range(win_size)],
        dtype=torch.float32,
    )
    win1D = (gauss / gauss.sum()).unsqueeze(1)
    window = (win1D @ win1D.T).expand(C, 1, win_size, win_size).contiguous().to(img1)
    half_win_size = win_size // 2

    mu1 = conv2d(img1, window, padding=half_win_size, groups=C)
    mu2 = conv2d(img2, window, padding=half_win_size, groups=C)

    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2

    var1 = conv2d(img1 * img1, window, padding=half_win_size, groups=C) - mu1_sq
    var2 = conv2d(img2 * img2, window, padding=half_win_size, groups=C) - mu2_sq
    var12 = conv2d(img1 * img2, window, padding=half_win_size, groups=C) - mu1_mu2

    k1, k2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + k1) * (2 * var12 + k2)) / (
        (mu1_sq + mu2_sq + k1) * (var1 + var2 + k2)
    )
    return ssim_map.mean()


def gaussian_splatting_loss(pred, target, ssim_lambda: float = 0.2):
    l1 = l1_loss(pred, target)
    _ssim = ssim(pred, target)
    loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - _ssim)
    return loss, l1.item(), _ssim.item()


def q_rotation_to_matrix(q: torch.Tensor):
    R = torch.zeros((len(q), 3, 3)).to(q)
    r, x, y, z = [q[:, i] for i in range(4)]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class TrainConfig:
    num_iterations: int = 30_000
    val_interval: int = 1_000
    sh_degree_update_interval: int = 1_000
    densify_start_iteration: int = 500
    densify_end_iteration: int = 15_000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    densify_split_clone_thres: float = 0.01
    densify_split_scale_factor: float = 1.6
    big_point_px_radius: int = 20
    big_point_cov_extent: float = 0.1
    min_opacity_pruning: float = 0.01
    max_num_gaussians: int = 3_000_000
    reset_opacity_interval: int = 3_000
    xyz_lr: float = 0.00016
    xyz_lr_scheduler_max_steps: float = 30_000
    shs_lr: float = 0.0025
    opacity_lr: float = 0.05
    scales_lr: float = 0.005
    rotations_lr: float = 0.001


class Densifier:
    def __init__(
        self,
        config: TrainConfig,
        gaussians: Gaussians,
        optimizer: torch.optim.Optimizer,
    ):
        self.config = config
        self.gaussians = gaussians
        self.optimizer = optimizer
        self._init_state()

    def _init_state(self):
        zeros_like = lambda x: torch.zeros(len(self.gaussians)).to(x)
        self.gradients = zeros_like(self.gradients)
        self.accumulated_iterations = zeros_like(self.accumulated_iterations)
        self.max_radii = zeros_like(self.max_radii)

    def _update_state(self, raster_outputs: RasterOutputs):
        mask = raster_outputs.frustum_mask
        n = self.accumulated_iterations[mask]
        new_grad = torch.norm(raster_outputs.screengrad_store.grad[mask], dim=-1)
        self.gradients[mask] = (n * self.gradients[mask] + new_grad) / (n + 1)
        self.accumulated_iterations[mask] += 1
        self.max_radii[mask] = torch.max(
            self.max_radii[mask], raster_outputs.radii[mask]
        )

    def _prune_parameters(self, mask: torch.Tensor):
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            new_param = torch.nn.Parameter(param[~mask], True).contiguous()
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][~mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][~mask]
                del self.optimizer.state[param]
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            setattr(self.gaussians, group["name"], new_param)

    def _add_parameters(self, parameters: dict):
        add_data = lambda x, y: torch.cat([x, y], dim=0)
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            added_params = parameters[group["name"]]
            new_param = torch.nn.Parameter(
                add_data(param, added_params), True
            ).contiguous()
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = add_data(
                    state["exp_avg"], torch.zeros_like(added_params)
                )
                state["exp_avg_sq"] = add_data(
                    state["exp_avg_sq"], torch.zeros_like(added_params)
                )
                del self.optimizer.state[param]
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            setattr(self.gaussians, group["name"], new_param)

    def _prune_gaussians(self, mask: torch.Tensor):
        self._prune_parameters(mask)
        self.accumulated_iterations = self.accumulated_iterations[~mask]
        self.gradients = self.gradients[~mask]
        self.max_radii = self.max_radii[~mask]

    def _add_gaussians(self, parameters: dict):
        self._add_parameters(parameters)
        add_offset = lambda x: torch.cat(
            [x, torch.zeros(len(self.gaussians) - len(x)).to(x)]
        )
        self.accumulated_iterations = add_offset(self.accumulated_iterations)
        self.gradients = add_offset(self.gradients)
        self.max_radii = add_offset(self.max_radii)

    def _clone(self):
        big_grad_mask = self.gradients >= self.config.densify_grad_threshold
        under_reconstruction_mask = (
            self.gaussians.sizes
            <= self.config.densify_split_clone_thres
            * self.gaussians.config.extent_radius
        )
        clone_mask = big_grad_mask & under_reconstruction_mask
        parameters = {
            n: getattr(self.gaussians, n)[clone_mask]
            for n in self.gaussians.param_names
        }
        self._add_gaussians(parameters)
        return clone_mask.sum().item()

    def _split(self):
        big_grad_mask = self.gradients >= self.config.densify_grad_threshold
        over_reconstruction_mask = (
            self.gaussians.sizes
            > self.config.densify_split_clone_thres
            * self.gaussians.config.extent_radius
        )
        split_mask = big_grad_mask & over_reconstruction_mask

        num_splits = 2
        scales = self.gaussians.scales[split_mask].repeat(num_splits, 1)
        samples = torch.normal(mean=torch.zeros(3).to(scales), std=scales)
        rotations = q_rotation_to_matrix(self.gaussians.q_rotations[split_mask]).repeat(
            num_splits, 1, 1
        )
        samples = rotations.bmm(samples.unsqueeze(-1)).squeeze(-1) + self.gaussians.xyz[
            split_mask
        ].repeat(num_splits, 1)

        parameters = {
            "xyz": samples,
            "colors": self.gaussians.colors[split_mask].repeat(num_splits, 1, 1),
            "sh_coeffs": self.gaussians.sh_coeffs[split_mask].repeat(num_splits, 1, 1),
            "logit_opacity": self.gaussians.logit_opacity[split_mask].repeat(
                num_splits, 1
            ),
            "log_scales": (
                self.gaussians.log_scales[split_mask]
                - math.log(self.config.densify_split_scale_factor)
            ).repeat(num_splits, 1),
            "quaternions": self.gaussians.quaternions[split_mask].repeat(num_splits, 1),
        }
        self._add_gaussians(parameters)

        prune_mask = torch.cat(
            [
                split_mask,
                torch.zeros(len(self.gaussians) - len(split_mask)).to(split_mask),
            ]
        )
        self._prune_gaussians(prune_mask)

        return split_mask.sum().item()

    def _prune_big_points(self):
        big_points_screen = self.max_radii > self.config.big_point_px_radius
        big_points_world = (
            self.gaussians.sizes
            > self.config.big_point_cov_extent * self.gaussians.extent_radius
        )
        self._prune_gaussians(big_points_screen | big_points_world)

    def _prune_opacity(self):
        prune_mask = self.gaussians.opacity.squeeze() < self.config.min_opacity_pruning
        self._prune_gaussians(prune_mask)

    def __call__(self, it: int, raster_outputs: RasterOutputs):
        num_clones, num_splits = 0, 0
        if it < self.config.densify_end_iteration:
            self._update_state(raster_outputs)
            if (
                it > self.config.densify_start_iteration
                and it % self.config.densify_interval == 0
            ):
                if len(self.gaussians) <= self.config.max_num_gaussians:
                    num_clones = self._clone()
                    num_splits = self._split()
                if it > self.config.reset_opacity_interval:
                    self._prune_big_points()
                self._prune_opacity()
                self._init_state()
                torch.cuda.empty_cache()
        return num_clones, num_splits


def reset_opacity(gaussians: Gaussians, optimizer: torch.optim.Optimizer):
    # sigmoid(-4.595) = 0.01
    new_logit_opacity = torch.clamp_max(gaussians.logit_opacity, -4.595)
    for group in optimizer.param_groups:
        if group["name"] == "logit_opacity":
            param = group["params"][0]
            new_param = torch.nn.Parameter(new_logit_opacity, True).contiguous()
            state = optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = torch.zeros_like(new_logit_opacity)
                state["exp_avg_sq"] = torch.zeros_like(new_logit_opacity)
                del optimizer.state[param]
                optimizer.state[new_param] = state
            group["params"][0] = new_param
            gaussians.logit_opacity = new_param
            return


def add_lr_update_method(obj):
    def update_lr(self, step):
        for p in self.param_groups:
            if "lr_scheduler" in p:
                p["lr"] *= p["lr_scheduler"](step)

    obj.update_lr = update_lr.__get__(obj)
    return obj


def test_data(gaussians: Gaussians, data: SceneData):
    val_l1 = [], val_ssim = []
    for j in range(len(data)):
        cam = data.get_camera(j, "cuda")
        img = gaussians.render(cam)
        _, l1, ssim = gaussian_splatting_loss(img / 255, data.rgb / 255)
        val_l1.append(l1), val_ssim.append(ssim)
    return statistics.mean(val_l1), statistics.mean(val_ssim)


def train(
    config: TrainConfig,
    gaussians: Gaussians,
    train_data: SceneData,
    val_data: SceneData = None,
):
    # define optimizer
    optimizer = add_lr_update_method(
        torch.optim.Adam(
            [
                {
                    "params": gaussians.xyz,
                    "lr": config.xyz_lr * gaussians.extent_radius,
                    "name": "xyz",
                    "lr_scheduler": lambda step: 0.01
                    ** min(step / config.xyz_lr_scheduler_max_steps, 1),
                },
                {"params": gaussians.colors, "lr": config.shs_lr, "name": "colors"},
                {
                    "params": gaussians.sh_coeffs,
                    "lr": config.shs_lr / 20,
                    "name": "sh_coeffs",
                },
                {
                    "params": gaussians.logit_opacity,
                    "lr": config.opacity_lr,
                    "name": "logit_opacity",
                },
                {
                    "params": gaussians.log_scales,
                    "lr": config.scales_lr,
                    "name": "log_scales",
                },
                {
                    "params": gaussians.quaternions,
                    "lr": config.rotations_lr,
                    "name": "quaternions",
                },
            ],
            eps=1e-15,
        )
    )

    logs = {
        k: [] for k in ["loss", "l1", "ssim", "clones", "splits", "val_l1", "val_ssim"]
    }
    sh_degree = 0
    densifier = Densifier(config, gaussians, optimizer)

    pbar = tqdm(range(1, config.num_iterations + 1), desc="Training gaussians")
    for i in pbar:
        log = {}
        cam_idx = random.randint(0, len(train_data) - 1)
        cam = train_data.get_camera(cam_idx, device="cuda")
        raster_outputs = gaussians.rasterize(cam, sh_degree=sh_degree)

        loss, log["l1"], log["ssim"] = gaussian_splatting_loss(
            raster_outputs.rgb, cam.rgb / 255.0
        )
        loss.backward()
        log["loss"] = loss.item()

        # update state
        if i % config.sh_degree_update_interval == 0:
            sh_degree = min(sh_degree + 1, gaussians.max_sh_degree)

        log["clones"], log["splits"] = densifier(i, raster_outputs)

        if (i % config.reset_opacity_interval == 0) or (
            gaussians.white_background and i == config.densify_end_iteration
        ):
            reset_opacity(gaussians)

        # update optimizers
        optimizer.step()
        optimizer.zero_grad()
        optimizer.update_lr(i)

        # logs
        if val_data is not None and i % config.val_interval == 0:
            log["val_l1"], log["val_ssim"] = test_data(gaussians, val_data)

        for k, v in log.items():
            logs[k].append(v)

        if i % 100 == 0:
            pbar.set_postfix(
                loss=statistics.mean(logs["loss"][-100:]),
                ssim=statistics.mean(logs["ssim"][-100:]),
                clones=sum(logs["clones"][-100:]),
                splits=sum(logs["splits"][-100:]),
                val_ssim=logs["val_ssim"][-1] if val_data is not None else None,
            )

        torch.cuda.empty_cache()

    return logs
