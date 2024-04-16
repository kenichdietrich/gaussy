# opt: colmap
# input: colmap folder + images folder

from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple
import random

import torch
from torchvision.io import read_image
from tqdm import tqdm
import pycolmap


Camera = namedtuple(
    "Camera", ["focal_length", "size", "position", "wc_matrix", "proj_matrix", "rgb"]
)

PointCloud = namedtuple("PointCloud", ["xyz", "color"])


@dataclass
class SceneData:
    focal_length: float
    image_size: tuple  # (W, H)
    camera_positions: torch.Tensor
    world_cam_matrices: torch.Tensor
    projection_matrices: torch.Tensor
    images: torch.Tensor
    point_cloud: PointCloud = None

    def __len__(self):
        return len(self.world_cam_matrices)

    def get_camera(self, idx, device: str = "cpu"):
        return Camera(
            self.focal_length,
            self.image_size,
            self.camera_positions[idx].to(device),
            self.world_cam_matrices[idx].to(device),
            self.projection_matrices[idx].to(device),
            self.images[idx].to(device),
        )

    def split(self, test_size: float = 0.2):
        num_cameras = len(self.world_cam_matrices)
        index = list(range(num_cameras))
        test_index = sorted(random.sample(index, int(num_cameras * test_size)))
        train_index = [i for i in index if i not in test_index]
        return SceneData(
            self.focal_length,
            self.image_size,
            self.camera_positions[train_index],
            self.world_cam_matrices[train_index],
            self.projection_matrices[train_index],
            self.images[train_index],
            self.point_cloud,
        ), SceneData(
            self.focal_length,
            self.image_size,
            self.camera_positions[test_index],
            self.world_cam_matrices[test_index],
            self.projection_matrices[test_index],
            self.images[test_index],
        )


def preprocess_scene(poses, point_cloud):
    # from opengl to opencv system
    poses[..., 1:3] *= -1

    # center to origin
    center = poses[:, :3, 3].mean(dim=0)
    poses[:, :3, 3] -= center
    xyz = point_cloud.xyz - center

    # scale cameras to 1 cube
    scale_factor = 1 / torch.max(poses[:, :3, 3].abs()).item()
    poses[:, :3, 3] *= scale_factor
    xyz *= scale_factor

    return poses, PointCloud(xyz, point_cloud.color)


def read_colmap(colmap_path: str | Path) -> tuple:
    reconstruction = pycolmap.Reconstruction(str(colmap_path))

    camera = reconstruction.cameras[1]
    focal_length = (camera.params[0] + camera.params[1]) / 2
    image_size = (camera.width, camera.height)

    images = reconstruction.images.values()
    poses = torch.stack(
        [
            torch.cat(
                [
                    torch.tensor(im.cam_from_world.inverse().matrix()),
                    torch.tensor([[0, 0, 0, 1.0]]),
                ],
                dim=0,
            ).to(torch.float32)
            for im in images
        ]
    )  # c2w
    image_names = [im.name for im in images]

    points = reconstruction.points3D.values()
    xyz = torch.stack([torch.from_numpy(p.xyz) for p in points]).to(torch.float32)
    colors = torch.stack([torch.from_numpy(p.color) for p in points]).to(torch.uint8)

    return focal_length, image_size, poses, image_names, PointCloud(xyz, colors)


def read_scene(scene_path: str | Path) -> SceneData:
    scene_path = Path(scene_path)
    focal_length, image_size, cam_world_matrices, image_names, point_cloud = (
        read_colmap(scene_path / "colmap" / "sparse" / "0")
    )
    cam_world_matrices, point_cloud = preprocess_scene(cam_world_matrices, point_cloud)

    world_cam_matrices = torch.zeros_like(cam_world_matrices)
    world_cam_matrices[:, :3, :3] = cam_world_matrices[:, :3, :3].transpose(1, 2)
    world_cam_matrices[:, :3, 3] = -torch.einsum(
        "bij,bj->bi",
        cam_world_matrices[:, :3, :3].transpose(1, 2),
        cam_world_matrices[:, :3, 3],
    )
    world_cam_matrices[:, 3, 3] = 1.0

    near, far = 0.01, 100.0
    frustum_matrix = torch.tensor(
        [
            [2 * focal_length / image_size[0], 0, 0, 0],
            [0, 2 * focal_length / image_size[1], 0, 0],
            [0, 0, far / (far - near), -far * near / (far - near)],
            [0, 0, 1.0, 0],
        ],
        dtype=torch.float32,
    )
    projection_matrices = frustum_matrix @ world_cam_matrices

    images = torch.stack(
        [
            read_image(str(scene_path / "images" / filename))
            for filename in tqdm(image_names, desc="Loading images")
        ]
    )
    return SceneData(
        focal_length,
        image_size,
        cam_world_matrices[:, :3, 3],
        world_cam_matrices,
        projection_matrices,
        images,
        point_cloud,
    )
