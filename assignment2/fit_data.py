import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from utils import get_mesh_renderer, get_points_renderer


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--output_path', default='images', type=str, required=False)
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')

def render_point_clouds(point_cloud, filename='point_cloud.jpg'):
    image_size = 512
    renderer = get_points_renderer(
        image_size=image_size
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    verts = torch.Tensor(point_cloud)
    # rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    # point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    color = (verts - verts.min()) / (verts.max() - verts.min())
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=color).to(device)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend[0, ..., :3].detach().cpu().numpy() # (B, H, W, 4) -> (H, W, 3)
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, filename)
    plt.imsave(output_path, rend)


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    render_point_clouds(pointclouds_src, filename='point_cloud_source.jpg')
    render_point_clouds(pointclouds_tgt, filename='point_cloud_target.jpg')
    print('Done!')

def render_voxels_tgt(voxels):
    voxels_forward_passed = voxels.squeeze(0).detach().cpu().numpy()
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels_forward_passed), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    return vertices, faces


def render_voxels(voxels):
    # voxels_forward_passed = voxels.squeeze(0).detach().cpu().numpy()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    voxels_forward_passed = torch.nn.Sigmoid()(voxels)
    voxels_forward_passed_rounded = torch.round(voxels_forward_passed, decimals=1).squeeze(0).detach().cpu().numpy() # decimal = 0 is safer, we set to 1 just to verify our fit
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels_forward_passed_rounded), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    voxel_size = voxels_forward_passed_rounded.shape[0]
    min_value = -1
    max_value = 1
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=512, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=120, azim=45)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, 'voxels_source.jpg')
    plt.imsave(output_path, rend)

def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    render_voxels(voxels_src)
    print('Done!')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
