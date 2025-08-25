# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import struct
import collections
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import platform

DEFAULT_DEVICE = torch.device("cuda")


def fov2focal(fov_radians: float, pixels: int):
    return pixels / (2 * math.tan(fov_radians / 2))


def focal2fov(focal: float, pixels: int):
    return 2 * math.atan(pixels / (2 * focal))


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None):
    """
    针孔相机模型的射线生成算法 - 透视投影的逆变换
    
    【核心原理】：
    这个函数实现了透视投影的逆变换，将2D像素坐标转换为3D射线方向。
    这是为什么每个像素都有不同射线方向的根本原因 - 这是透视投影的本质！
    
    【数学推导】：
    正向透视投影公式：
        screen_x = (world_x / world_z) * focal_x + principal_x
        screen_y = (world_y / world_z) * focal_y + principal_y
    
    逆向求射线方向：
        world_x/world_z = (screen_x - principal_x) / focal_x = xs
        world_y/world_z = (screen_y - principal_y) / focal_y = ys  
        world_z/world_z = 1
    
    所以射线方向就是 normalize((xs, ys, 1))！
    
    【为什么不是所有射线都平行于相机朝向？】：
    - 只有图像中心的像素射线方向才是相机朝向 (0, 0, 1)
    - 其他像素都有偏移，创造透视效果 (近大远小)
    - 如果所有射线都平行，就变成正交投影，失去透视效果
    
    【实际计算示例】(假设 1024×768, fx=fy=800):
        像素(0,0):     xs=-0.64, ys=-0.48 → 方向(-0.54, -0.41, 0.85) 朝左上
        像素(512,384): xs=0,     ys=0     → 方向(0, 0, 1)           相机朝向  
        像素(1023,767): xs=0.64,  ys=0.48  → 方向(0.54, 0.41, 0.85)  朝右下
    
    【与GPU内部计算的对比】：
    本函数(外部预计算):   xs = (x - w/2) / fx
    GPU内部计算:         dir_x = (x - w/2) * tanFoV  
    数学本质相同：tanFoV ≈ 1/fx (小角度近似)
    
    Args:
        x, y: 像素坐标数组
        f_x, f_y: 焦距参数 (像素单位)
        w, h: 图像宽度和高度
        ray_jitter: 可选的射线抖动，用于抗锯齿
    
    Returns:
        ray_origin (sz_y, sz_x, 3): 射线起点 (相机空间原点)
        normalized ray_direction (sz_y, sz_x, 3): 归一化的射线方向 (每个像素不同!)
    """

    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).numpy()
        jitter_xs = jitter[:, 0]
        jitter_ys = jitter[:, 1]
    else:
        jitter_xs = jitter_ys = 0.5

    # 🎯 核心公式：透视投影逆变换
    # 将像素坐标转换为相机空间的标准化坐标
    # principal_point = (w/2, h/2) - 图像中心为主点
    xs = ((x + jitter_xs) - 0.5 * w) / f_x  # X方向标准化：(像素X - 图像中心X) / 焦距X
    ys = ((y + jitter_ys) - 0.5 * h) / f_y  # Y方向标准化：(像素Y - 图像中心Y) / 焦距Y

    # 构造射线方向向量：(xs, ys, 1)
    # 为什么Z=1？因为相机空间中，成像平面位于Z=1处
    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    
    # 射线起点都是相机位置（相机空间原点）
    ray_origin = np.zeros_like(ray_lookat)

    # 归一化射线方向，确保是单位向量
    # 这样每个像素都得到了唯一的射线方向，实现正确的透视投影！
    return ray_origin, ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def camera_to_world_rays(ray_o, ray_d, poses):
    """
    相机空间到世界空间的射线变换
    
    【功能】：
    将 pinhole_camera_rays() 生成的相机空间射线转换到世界坐标系。
    这是射线预计算流程的第二步，确保射线在正确的世界位置和朝向。
    
    【数学原理】：
    射线变换使用标准的齐次坐标变换：
        world_point = R * camera_point + t
        world_direction = R * camera_direction
    
    其中：
    - R: 3×3 旋转矩阵 (poses[:, :3, :3])
    - t: 3×1 平移向量 (poses[:, :3, 3])
    
    【为什么方向不需要平移？】：
    - 位置需要平移：相机原点 (0,0,0) → 世界空间的相机位置
    - 方向只需旋转：方向向量描述朝向，不受位置影响
    
    【在渲染管线中的作用】：
    经过这个变换后，每个像素都有了：
    1. 世界空间的射线起点 (通常是相机在世界中的位置)
    2. 世界空间的射线方向 (每个像素都不同，保持透视效果)
    
    【数据流】：
    pinhole_camera_rays() → 相机空间射线 → camera_to_world_rays() → 世界空间射线 → 渲染器
    
    Args:
        ray_o_cam [n, 3]: 相机坐标系下的射线起点
        ray_d_cam [n, 3]: 相机坐标系下的射线方向  
        poses [n, 4, 4]: 相机到世界的变换矩阵 (包含旋转和平移)

    Returns:
        ray_o [n, 3]: 世界坐标系下的射线起点
        ray_d [n, 3]: 世界坐标系下的射线方向 (每个像素仍然不同!)
    """
    if isinstance(poses, torch.Tensor):
        # 射线起点变换: world_origin = R * camera_origin + t
        ray_o = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        # 射线方向变换: world_direction = R * camera_direction (只旋转，不平移)
        ray_d = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)
    else:
        # NumPy 版本的相同计算
        ray_o = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        ray_d = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)

    return ray_o, ray_d


@dataclass(slots=True, kw_only=True)
class PointCloud:
    """Represents a 3d point cloud consisting of corresponding start and end points"""

    xyz_start: torch.Tensor  # [N,3]
    xyz_end: torch.Tensor  # [N,3]
    device: str
    dtype = torch.float32
    color: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert len(self.xyz_start) == len(self.xyz_end)
        assert self.xyz_start.shape[1] == self.xyz_end.shape[1] == 3

        self.xyz_start.to(self.device, dtype=self.dtype)
        self.xyz_end.to(self.device, dtype=self.dtype)

        if self.color is not None:
            assert self.color.shape[1] == 3
            assert len(self.color) == len(self.xyz_end)

            self.color.to(self.device, dtype=self.dtype)

    @staticmethod
    def from_sequence(point_clouds: Sequence[PointCloud], device: str) -> PointCloud:
        point_clouds_list = list(point_clouds)

        return PointCloud(
            xyz_start=torch.cat([pc.xyz_start for pc in point_clouds_list]),
            xyz_end=torch.cat([pc.xyz_end for pc in point_clouds_list]),
            color=(
                torch.cat([pc.color for pc in point_clouds_list])
                if point_clouds_list[0].color is not None
                else None
            ),
            device=device,
        )

    def selected_idxs(self, idxs):
        return PointCloud(
            xyz_start=self.xyz_start[idxs],
            xyz_end=self.xyz_end[idxs],
            color=self.color[idxs] if self.color is not None else None,
            device=self.device,
        )


def get_center_and_diag(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def compute_max_distance_to_border(
    image_size_component: float, principal_point_component: float
) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary.
    """
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


def create_camera_visualization(cam_list):
    """
    Given a list-of-dicts of camera & image info, register them in polyscope
    to create a visualization
    """

    import polyscope as ps

    for i_cam, cam in enumerate(cam_list):

        ps_cam_param = ps.CameraParameters(
            ps.CameraIntrinsics(
                fov_vertical_deg=np.degrees(cam["fov_h"]),
                fov_horizontal_deg=np.degrees(cam["fov_w"]),
            ),
            ps.CameraExtrinsics(mat=cam["ext_mat"]),
        )

        cam_color = (1.0, 1.0, 1.0)
        if cam["split"] == "train":
            cam_color = (1.0, 0.7, 0.7)
        elif cam["split"] == "val":
            cam_color = (0.7, 0.1, 0.7)

        ps_cam = ps.register_camera_view(
            f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color
        )

        ps_cam.add_color_image_quantity(
            "target image", cam["rgb_img"][:, :, :3], enabled=True
        )


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_points3D_text(path):
    """
    Read points3D.txt file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    # Single file read
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # Convert directly to numpy arrays while appending
                xyzs.append([float(x) for x in elems[1:4]])
                rgbs.append([int(x) for x in elems[4:7]])
                errors.append(float(elems[7]))

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_points3D_binary(path_to_model_file):
    """
    Read points3D.bin file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            # Read the point data
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            # Append coordinates, colors, and error
            xyzs.append(binary_point_line_properties[1:4])
            rgbs.append(binary_point_line_properties[4:7])
            errors.append(binary_point_line_properties[7])

            # Skip track length and elements as they're not used
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            fid.seek(8 * track_length, 1)

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_intrinsics_text(path):
    """
    Read camera intrinsics from a COLMAP text file.
    Args:
        path: Path to the cameras.txt file
    Returns:
        Dict of Camera objects indexed by camera ID
    """
    cameras = {}
    with open(path, "r") as fid:
        # Skip comment lines at the start
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        for line in lines:
            # Unpack elements directly using split with maxsplit
            camera_id, model, width, height, *params = line.split()
            camera_id = int(camera_id)
            width, height = int(width), int(height)
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=np.array([float(p) for p in params]),
            )
    return cameras


def read_colmap_intrinsics_binary(path_to_model_file):
    """
    Read camera intrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the cameras.bin file
    Returns:
        Dict of Camera objects indexed by camera ID
    Raises:
        ValueError: If the number of cameras read doesn't match the expected count
        KeyError: If an invalid camera model ID is encountered
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # Read fixed-size camera properties
            camera_id, model_id, width, height = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            # Get camera model information
            try:
                camera_model = CAMERA_MODEL_IDS[model_id]
            except KeyError:
                raise KeyError(f"Invalid camera model ID: {model_id}")
            # Read camera parameters
            params = read_next_bytes(
                fid,
                num_bytes=8 * camera_model.num_params,
                format_char_sequence="d" * camera_model.num_params,
            )
            # Create camera object
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model.model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    # Verify camera count
    assert (
        len(cameras) == num_cameras
    ), f"Expected {num_cameras} cameras, but read {len(cameras)}"
    return cameras


def qvec_to_so3(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec_to_so3(self):
        return qvec_to_so3(self.qvec)


def read_colmap_extrinsics_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the images.bin file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If string parsing or data reading fails
    """
    images = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of registered images
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # Read image properties (id, rotation, translation, camera_id)
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id, *qvec_tvec, camera_id = props
            qvec = np.array(qvec_tvec[:4])
            tvec = np.array(qvec_tvec[4:7])
            # Read image name (null-terminated string)
            image_name = ""
            while True:
                current_char = read_next_bytes(fid, 1, "c")[0]
                if current_char == b"\x00":
                    break
                try:
                    image_name += current_char.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(
                        f"Invalid character in image name at position {len(image_name)}"
                    )
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            point_data = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            # Parse point data into coordinates and IDs
            xys = np.array(
                [
                    (point_data[i], point_data[i + 1])
                    for i in range(0, len(point_data), 3)
                ]
            )
            point3D_ids = np.array(
                [int(point_data[i + 2]) for i in range(0, len(point_data), 3)]
            )
            # Create image object
            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return sorted(images, key=lambda x: x.name)


def read_colmap_extrinsics_text(path):
    """
    Read camera extrinsics from a COLMAP text file.
    Args:
        path: Path to the images.txt file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If file format is invalid or data parsing fails
    """
    images = []
    with open(path, "r") as fid:
        # Skip comment lines and get valid lines
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        # Process lines in pairs (image info + points info)
        try:
            while True:
                # Read image info line
                image_line = next(lines, None)
                if image_line is None:
                    break
                # Parse image properties
                elems = image_line.split()
                if len(elems) < 10:  # Minimum required elements
                    raise ValueError(f"Invalid image line format: {image_line}")
                image_id = int(elems[0])
                qvec = np.array([float(x) for x in elems[1:5]])
                tvec = np.array([float(x) for x in elems[5:8]])
                camera_id = int(elems[8])
                image_name = elems[9]
                # Read points line
                points_line = next(lines, None)
                if points_line is None:
                    raise ValueError(f"Missing points data for image {image_name}")
                # Parse 2D points and 3D point IDs
                point_elems = points_line.split()
                if len(point_elems) % 3 != 0:
                    raise ValueError(f"Invalid points format for image {image_name}")
                xys = np.array(
                    [
                        (float(point_elems[i]), float(point_elems[i + 1]))
                        for i in range(0, len(point_elems), 3)
                    ]
                )
                point3D_ids = np.array(
                    [int(point_elems[i + 2]) for i in range(0, len(point_elems), 3)]
                )
                # Create image object
                images.append(
                    Image(
                        id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        name=image_name,
                        xys=xys,
                        point3D_ids=point3D_ids,
                    )
                )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing extrinsics file: {e}")
    return sorted(images, key=lambda x: x.name)


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader multiprocessing.

    This function ensures that each worker process has a proper CUDA context
    and random number generator state, which is especially important on Windows.
    """
    import random
    import numpy as np
    import torch

    # Set random seeds for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Initialize CUDA context in worker process
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
        # Force CUDA context creation by doing a small operation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def configure_dataloader_for_platform(dataloader_kwargs: dict) -> dict:
    """
    Configure DataLoader kwargs for the current platform.

    Args:
        dataloader_kwargs: Dictionary of DataLoader arguments
        force_windows_multiprocessing: If True, allow multiprocessing on Windows despite potential issues

    Returns:
        Updated DataLoader kwargs
    """
    kwargs = dataloader_kwargs.copy()

    if "num_workers" in kwargs:
        original_num_workers = kwargs["num_workers"]
        kwargs["num_workers"] = original_num_workers

        # Adjust persistent_workers based on actual num_workers
        if "persistent_workers" in kwargs:
            kwargs["persistent_workers"] = kwargs["num_workers"] > 0

        # On Windows with multiprocessing, add worker initialization function
        if platform.system() == "Windows" and kwargs["num_workers"] > 0:
            kwargs["worker_init_fn"] = worker_init_fn

    return kwargs


def get_worker_id():
    """Get current worker ID for thread-local caching."""
    import threading

    # Get worker ID from current process/thread
    try:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return f"worker_{worker_info.id}"
        else:
            return "main_process"
    except:
        return f"thread_{threading.get_ident()}"