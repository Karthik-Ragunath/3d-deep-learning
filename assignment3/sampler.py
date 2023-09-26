import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from torch.distributions.uniform import Uniform


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def get_device(self):
        """
        Checks if GPU is available and returns device accordingly.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return device

    def forward(
        self,
        ray_bundle,
    ):
        device = self.get_device()
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = Uniform(self.min_depth, self.max_depth).sample((self.n_pts_per_ray,1)).unsqueeze(0).repeat(ray_bundle.directions.shape[0], 1, 1).to(device)
        restructured_ray_directions = ray_bundle.directions.view(ray_bundle.directions.shape[0], -1, 3)
        # TODO (1.4): Sample points from z values # N * 64 * 3
        ray_origins_reshaped = ray_bundle.origins.view(-1, 1, 3)
        sample_points = ray_origins_reshaped + torch.bmm(z_vals, restructured_ray_directions)
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )
    
    # def forward(
    #     self,
    #     ray_bundle,
    # ):
    #     # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
    #     z_vals = torch.linspace(
    #         self.min_depth, self.max_depth, self.n_pts_per_ray).view(1, -1, 1).cuda()
    #     N = ray_bundle.directions.shape[0]
    #     d = z_vals.shape[1]
    #     z_vals = z_vals.repeat(N, 1, 1)

    #     # TODO (1.4): Sample points from z values
    #     sample_points = z_vals * \
    #         ray_bundle.directions.view(-1, 1, 3) + \
    #         ray_bundle.origins.view(-1, 1, 3)

    #     # Return
    #     return ray_bundle._replace(
    #         sample_points=sample_points,
    #         sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
    #     )
    
    # def forward(
    #         self,
    #         ray_bundle,
    #     ):
        

    #     # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
    #     # Generates a set of distances between near and far
    #     device = self.get_device()
    #     z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to(device)
    #     z_vals = z_vals.unsqueeze(0)
    #     z_vals = z_vals.unsqueeze(2)
    #     # print(f" z_vals shape: {z_vals.shape} \n")
    #     N = ray_bundle.directions.shape[0]
 
    #     # z_vals = z_vals.repeat(N,1,1)


    #     # TODO (1.4):  Uses these distances to sample points offset from ray origins (RayBundle.origins) along ray directions (RayBundle.directions). 
    #     # print(f"z_vals shape: {z_vals.shape}, ray_bundle.directions shape: {ray_bundle.directions.shape}, ray_bundle.origins shape: {ray_bundle.origins.shape}")
    #     ray_bundle_reshape = ray_bundle.directions.view(-1, 1, 3)
    #     ray_bundle_or_reshape = ray_bundle.origins.view(-1, 1, 3)
    #     # print(f" shape of z_vals: {z_vals.shape} \n")
    #     # print(f" shape of ray_bundle_reshape: {ray_bundle_reshape.shape} \n")
    #     # print(f" shape of ray_bundle_or_reshape: {ray_bundle_or_reshape.shape} \n")
    #     sample_points = z_vals * ray_bundle.directions.view(-1, 1, 3) + ray_bundle.origins.view(-1, 1, 3)
    #     # print(f"sample_points shape: {sample_points.shape}")
    #     # print(f"sample_points: {sample_points}")
    #     # sys.exit()

     

    #     # Stores the distances and sample points in RayBundle.sample_points and RayBundle.sample_lengths
    #     # Return
    #     return ray_bundle._replace(
    #         sample_points=sample_points,
    #         sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
    #     )


sampler_dict = {
    'stratified': StratifiedRaysampler
}