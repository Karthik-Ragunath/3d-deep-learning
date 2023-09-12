"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer
from pytorch3d.renderer.cameras import look_at_view_transform
import os

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/multiview")
    args = parser.parse_args()

    # plt.imsave(args.output_path, render_cow(cow_path=args.cow_path, image_size=args.image_size))
    os.makedirs('images/multiview/', exist_ok=True)
    render_cow_360(cow_path=args.cow_path, image_size=args.image_size, output_dir=args.output_path)
    print("done")
