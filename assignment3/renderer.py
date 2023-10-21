import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def get_device(self):
        """
        Checks if GPU is available and returns device accordingly.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return device

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # # TODO (1.5): Compute transmittance using the equation described in the README
        # T_list = []
        # T = 1
        # for delta, rays_density_ele in zip(deltas, rays_density):
        #     T_list.append(T * (torch.exp(-delta * rays_density_ele)))
        #     T = T_list[-1]
        # T_tensor = torch.stack(T_list, dim=0) # torch.Size([32768, 64, 1])
        # # TODO (1.5): Compute weight used for rendering from transmittance and density
        # rendering_weights = T_tensor * (1 - torch.exp(-rays_density * deltas + eps)) # torch.Size([32768, 64, 1])
        # return rendering_weights
    
        num_rays = rays_density.shape[0]
        num_samples = rays_density.shape[1]

        T_prev = torch.ones(num_rays, 1).to(self.get_device())
  

        T = []
        for i in range(num_samples):
            T_curr = T_prev * torch.exp(-rays_density[:, i] * deltas[:, i])
            T.append(T_curr)
            T_prev = T_curr

        T = torch.stack(T, dim=1)
        transmittance = (1 - torch.exp(-rays_density * deltas + eps))
        # print(f"shape of T {T.shape}, transmittance {transmittance.shape}")
        weights = T * transmittance

        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor, # torch.Size([32768, 64, 1])
        rays_feature: torch.Tensor # torch.Size([2097152, 3])
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        ray_features_reshaped = rays_feature.view(weights.shape[0], weights.shape[1], -1) # torch.Size([32768, 64, 3])
        feature = torch.sum(weights * ray_features_reshaped, dim=1)
        return feature
        # chunk_size = weights.shape[0]
        # num_samples = weights.shape[1]
        # rays_feature_reshape = rays_feature.view(chunk_size, num_samples, -1)
        # # rays_feature_reshape = rays_feature_reshape.squeeze(2)
        # # print(f"rays_feature_reshape shape: {rays_feature_reshape.shape}, weights shape: {weights.shape}")
        # feature = torch.sum(weights * rays_feature_reshape, dim=1)
        # return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1] # 64

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density'] # torch.Size([2097152, 1])
            feature = implicit_output['feature'] # torch.Size([2097152, 3])

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0] # torch.Size([32768, 64])
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None] # torch.Size([32768, 64, 1])

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1), # torch.Size([32768, 64, 1])
                density.view(-1, n_pts, 1) # torch.Size([32768, 64, 1])
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights=weights, rays_feature=feature) # torch.Size([32768, 3])

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights=weights, rays_feature=depth_values) # torch.Size([32768, 1])

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
