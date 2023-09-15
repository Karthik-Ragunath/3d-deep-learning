"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from pytorch3d.structures import Meshes
from starter.utils import get_device, get_mesh_renderer
from pytorch3d.renderer.cameras import look_at_view_transform
import os
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.io import load_objs_as_meshes, load_obj

def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

def render_cow_multi_camera_views(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
    angle=0,
    axis="X"
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    verts, faces, aux = load_obj(cow_path)
    faces_idx = faces.verts_idx.to(device)
    angle = torch.tensor(angle)
    
    '''
    homogeneous_matrix = RotateAxisAngle(angle, axis=axis, degrees=True).get_matrix()
    R_relative = homogeneous_matrix[:, :3, :3]
    R_relative = R_relative.squeeze(0)
    T_relative = homogeneous_matrix[:, :3, 3].squeeze(0)
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    T = R_relative @ torch.tensor([0.0, 0.0, 3.0]) + T_relative
    '''
    
    homogeneous_matrix = RotateAxisAngle(angle, axis=axis, degrees=True, device=device)
    verts = homogeneous_matrix.transform_points(meshes.verts_list()[0])
    meshes = Meshes(verts=[verts], faces=[faces_idx], textures=meshes.textures)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative

    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

def render_cow_360(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
    output_dir=''
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    azimuth_angle = 45
    for azimuth_angle in range(0, 361):
        R, T = look_at_view_transform(dist=5, elev=30, azim=azimuth_angle)
        renderer = get_mesh_renderer(image_size=image_size)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, device=device,
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
        rend = renderer(meshes, cameras=cameras, lights=lights)
        output_path = os.path.join(output_dir, f"{azimuth_angle}.jpg")
        plt.imsave(output_path, rend[0, ..., :3].cpu().numpy())

def render_cow_360_texture_modified(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
    output_dir=''
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    color_1 = torch.tensor([0, 0, 1]).to(device)
    color_2 = torch.tensor([1, 0, 0]).to(device)
    vertices = meshes.verts_list()[0]
    z_min = torch.min(vertices[:, 2])
    z_max = torch.max(vertices[:, 2])
    alpha = (vertices[:, 2] - z_min) / (z_max - z_min)
    colors = alpha.view(-1, 1) * color_1.view(1, 3) + (1 - alpha.view(-1, 1)) * color_2.view(1, 3)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    azimuth_angle = 45
    meshes.textures = textures
    for azimuth_angle in range(0, 361):
        R, T = look_at_view_transform(dist=5, elev=30, azim=azimuth_angle)
        renderer = get_mesh_renderer(image_size=image_size)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, device=device,
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
        rend = renderer(meshes, cameras=cameras, lights=lights)
        output_path = os.path.join(output_dir, f"{azimuth_angle}.jpg")
        plt.imsave(output_path, rend[0, ..., :3].cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/multiview")
    parser.add_argument("--camera_transform", action="store_true", required=False)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.camera_transform:
        angles = [(60.0, "X"), (60.0, "Y"), (90.0, "Z"), (90.0, "Y"), (90.0, "X"), (60.0, "Z"), (270.0, "X"), (270.0, "Y"), (270.0, "Z")]
        for angle, axis in angles:
            plt.imsave(os.path.join(args.output_path, f"{int(angle)}_{axis}.jpg"), render_cow_multi_camera_views(cow_path=args.cow_path, image_size=args.image_size, angle=angle, axis=axis))
    else:
        render_cow_360_texture_modified(cow_path=args.cow_path, image_size=args.image_size, output_dir=args.output_path)
        print("done")
