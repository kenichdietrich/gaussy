from pathlib import Path

import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from diff_gaussian_rasterization import (
    _RasterizeGaussians,
    GaussianRasterizationSettings,
)
from collections import namedtuple

from .data import SceneData, Camera

RasterOutputs = namedtuple(
    "RasterOutputs", ["rgb", "radii", "frustum_mask", "screengrad_store"]
)


def rasterize_gaussians(xyz, shs, opacities, scales, q_rotations, raster_settings):
    screengrad_store = torch.zeros_like(
        xyz[:, :2], requires_grad=True
    )  # screen gradient store
    screengrad_store.retain_grad()
    rgb, radii = _RasterizeGaussians.apply(
        means3D=xyz,
        means2D=screengrad_store,
        sh=shs,
        colors_precomp=torch.Tensor([]),
        opacities=opacities,
        scales=scales,
        rotations=q_rotations,
        cov3Ds_precomp=torch.Tensor([]),
        raster_settings=raster_settings,
    )
    return RasterOutputs(rgb, radii, radii > 0, screengrad_store)


def get_extent_radius(xyz: torch.Tensor) -> float:
    center = xyz.mean(dim=0)
    dists = torch.norm(xyz - center, dim=1)
    return dists.max().item() * 1.1


class Gaussians(nn.Module):
    param_names = [
        "xyz",
        "colors",
        "sh_coeffs",
        "logit_opacity",
        "log_scales",
        "quaternions",
    ]

    def __init__(self, max_sh_degree: int = 3, white_background: bool = True):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.num_sh_coeffs = (max_sh_degree + 1) ** 2 - 1
        self.white_background = white_background
        self.register_buffer(
            "background", torch.ones(3) if white_background else torch.zeros(3)
        )
        self.extent_radius = 1.0

    def __len__(self):
        return len(self.xyz)

    @property
    def shs(self):
        return torch.cat(
            [
                self.colors,
                self.sh_coeffs,
            ],
            dim=1,
        ).contiguous()

    @property
    def opacity(self):
        return torch.sigmoid(self.logit_opacity)

    @property
    def scales(self):
        return torch.exp(self.log_scales)

    @property
    def q_rotations(self):
        return nn.functional.normalize(self.quaternions)

    @property
    def sizes(self):
        return torch.max(self.scales, dim=-1).values

    def set_parameters(self, state_dict: dict):
        for k, v in state_dict.items():
            setattr(self, k, nn.Parameter(v, requires_grad=True).contiguous())

    def load_scene(self, data: SceneData):
        pc = data.point_cloud
        kdtree = cKDTree(pc.xyz.numpy())
        nn_distances = torch.from_numpy(kdtree.query(pc.xyz.numpy(), k=1 + 3)[0])
        nn_radius = torch.clamp_min(nn_distances[:, 1:].mean(dim=-1), 1e-6)

        parameters = {
            "xyz": pc.xyz,
            "colors": (pc.color.unsqueeze(-2) / 255 - 0.5) / 0.28209479177387814,
            "sh_coeffs": torch.zeros(len(pc.xyz), self.num_sh_coeffs, 3),
            "logit_opacity": torch.logit(0.1 * torch.ones_like(pc.xyz[:, :1])),
            "log_scales": torch.log(nn_radius).unsqueeze(-1).repeat(1, 3),
            "quaternions": torch.cat(
                [torch.ones_like(pc.xyz[:, :1]), torch.zeros_like(pc.xyz)], dim=-1
            ),
        }
        self.set_parameters(parameters)
        self.extent_radius = get_extent_radius(data.camera_positions)
        return self

    def load_ply(self, ply_path: str | Path):
        data = PlyData.read(str(ply_path))["vertex"].data
        stack_data = lambda varnames: torch.stack(
            [torch.from_numpy(data[n]) for n in varnames], dim=-1
        )
        parameters = {
            "xyz": stack_data("xyz"),
            "colors": stack_data(f"f_dc_{i}" for i in "012").unsqueeze(-2),
            "sh_coeffs": stack_data(n for n in data.dtype.names if "f_rest" in n)[
                :, : 3 * self.num_sh_coeffs
            ].reshape(-1, self.num_sh_coeffs, 3),
            "logit_opacity": torch.from_numpy(data["opacity"]).unsqueeze(-1),
            "log_scales": stack_data(f"scale_{i}" for i in "012"),
            "quaternions": stack_data(f"rot_{i}" for i in "0123"),
        }
        self.set_parameters(parameters)
        self.extent_radius = get_extent_radius(parameters["xyz"])
        return self

    def save_ply(self, ply_path: str | Path):
        varnames = (
            list("xyz")
            + ["nx", "ny", "nz"]
            + [f"f_dc_{i}" for i in "012"]
            + [f"f_rest_{i}" for i in range(3 * self.num_sh_coeffs)]
            + ["opacity"]
            + [f"scale_{i}" for i in "012"]
            + [f"rot_{i}" for i in "0123"]
        )
        data = (
            torch.cat(
                [
                    self.xyz,
                    torch.zeros_like(self.xyz),
                    self.colors.squeeze(),
                    self.sh_coeffs.reshape(-1, 3 * self.num_sh_coeffs),
                    self.logit_opacity,
                    self.log_scales,
                    self.q_rotations,
                ],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )
        data.dtype = [(n, "f4") for n in varnames]
        elements = PlyElement.describe(data.flatten(), "vertex")
        PlyData([elements]).write(str(ply_path))

    @property
    def device(self):
        return next(self.parameters()).device

    def rasterize(
        self, camera: Camera, scale_modifier: float = 1.0, sh_degree: int = 3
    ) -> RasterOutputs:
        assert self.device.type == "cuda", "gaussians must be on a CUDA device!"
        assert (
            self.device == camera.wc_matrix.device
        ), "camera is not on the gaussians' device!"
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_size[1],
            image_width=camera.image_size[0],
            tanfovx=0.5 * camera.image_size[0] / camera.focal_length,
            tanfovy=0.5 * camera.image_size[1] / camera.focal_length,
            bg=self.background,
            scale_modifier=scale_modifier,
            viewmatrix=camera.wc_matrix.T,
            projmatrix=camera.proj_matrix.T,
            sh_degree=sh_degree,
            campos=camera.position,
            prefiltered=False,
            debug=False,
        )
        outputs = rasterize_gaussians(
            self.xyz,
            self.shs,
            self.opacity,
            self.scales,
            self.q_rotations,
            raster_settings,
        )
        return outputs

    @torch.no_grad()
    def render(self, camera: Camera):
        raster_outputs = self.rasterize(camera)
        rgb = (raster_outputs.rgb.clip(0.0, 1.0) * 255).round()
        return rgb.to(torch.uint8)
