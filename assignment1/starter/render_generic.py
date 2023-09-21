"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus(image_size=256, num_samples=200, device=None, output_path="images/parametric_torus.jpg"):
    """Render torus."""
    if device is None:
        device = get_device()
    theta = torch.linspace(0, 2 * np.pi, num_samples) # angle of rotation around central axis
    phi = torch.linspace(0, 2 * np.pi, num_samples) # angle of rotation within cross section
    Phi, Theta = torch.meshgrid(phi, theta)
    # Assuming R=4, r=1
    R = 1
    r = 0.5
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.sin(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud, cameras=cameras)
    rend = rend[0, ..., :3].cpu().numpy()
    plt.imsave(output_path, rend)


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_torus_mesh(image_size=256, voxel_size=64, device=None, output_path="images/implicit_torus.jpg"):
    if device is None:
        device = get_device()
    min_value = -1.0
    max_value = 1.0
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = torch.pow((torch.pow((X ** 2 + Y ** 2), 0.5) - 1 ** 2), 2) + Z ** 2 - 0.25 ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    plt.imsave(output_path, rend)

def render_rgdb(rgbd_dict: dict):
    rgb_1 = rgbd_dict.get('rgb1')
    rgb_2 = rgbd_dict.get('rgb2')
    mask_1 = rgbd_dict.get('mask1')
    mask_2 = rgbd_dict.get('mask2')
    depth_1 = rgbd_dict.get('depth1')
    depth_2 = rgbd_dict.get('depth2')
    camera_1 = rgbd_dict.get('cameras1')
    camera_2 = rgbd_dict.get('cameras2')
    image_size = rgb_1.shape[0]
    points_1, rgb_1_depth = unproject_depth_image(rgb_1, mask_1, depth_1, camera_1)
    points_2, rgb_2_depth = unproject_depth_image(rgb_2, mask_2, depth_2, camera_2)
    device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=(1, 1, 1)
    )
    verts_1 = points_1.to(device)
    rgbd_1 = rgb_1_depth.to(device)

    verts_2 = points_2.to(device)
    rgbd_2 = rgb_2_depth.to(device)
    
    R, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=10, azim=0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    point_cloud_1 = pytorch3d.structures.Pointclouds(points=verts_1.unsqueeze(0), features=rgbd_1.unsqueeze(0))
    rend_1 = renderer(point_cloud_1, cameras=cameras)
    rend_1 = rend_1.cpu().numpy()[0, ..., :3]
    plt.imsave("images/point_cloud_1.jpg", rend_1)

    point_cloud_2 = pytorch3d.structures.Pointclouds(points=verts_2.unsqueeze(0), features=rgbd_2.unsqueeze(0))
    rend_2 = renderer(point_cloud_2, cameras=cameras)
    rend_2 = rend_2.cpu().numpy()[0, ..., :3]
    plt.imsave("images/point_cloud_2.jpg", rend_2)

    point_cloud_3 = pytorch3d.structures.Pointclouds(points=torch.cat((verts_1, verts_2), dim=0).unsqueeze(0), features=torch.cat((rgbd_1, rgbd_2), dim=0).unsqueeze(0))
    rend_3 = renderer(point_cloud_3, cameras=cameras)
    rend_3 = rend_3.cpu().numpy()[0, ..., :3]
    plt.imsave("images/point_cloud_3.jpg", rend_3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "check_loaded_data", "render_rgbd", "parametric_torus", "implicit_torus"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "parametric_torus":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples, output_path=args.output_path)
        exit(0)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "implicit_torus":
        render_torus_mesh(image_size=args.image_size, output_path=args.output_path)
        exit(0)
    elif args.render == "check_loaded_data":
        rgbd_image_data = load_rgbd_data()
        exit(0)
    elif args.render == "render_rgbd":
        rgbd_image_data = load_rgbd_data()
        render_rgdb(rgbd_dict=rgbd_image_data)
        exit(0)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

