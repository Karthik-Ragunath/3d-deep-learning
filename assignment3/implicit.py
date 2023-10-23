import torch
import torch.nn.functional as F

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)

        return torch.linalg.norm(
            sample_points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3) # torch.Size([2097152, 3])
        diff = torch.abs(sample_points - self.center) - self.side_lengths / 2.0 # torch.Size([2097152, 3])
        # self.center
        # Parameter containing:
        # tensor([[0., 0., 0.]], device='cuda:0', requires_grad=True)
        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)), # torch.Size([2097152, 3])
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0])) # torch.max(diff, dim=-1)[0] -> torch.Size([2097152])
        
        # signed_distance.shape - torch.Size([2097152])
        
        # torch.min(min_vector)
        # tensor(-0.8434, device='cuda:0', grad_fn=<MinBackward1>)
        # torch.max(min_vector)
        # tensor(0., device='cuda:0', grad_fn=<MaxBackward1>)

        return signed_distance.unsqueeze(-1) # torch.Size([2097152, 1])


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3) # torch.Size([2097152, 3])
        depth_values = ray_bundle.sample_lengths[..., 0] # torch.Size([32768, 64])
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1) # torch.Size([2097152, 1])

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle) # torch.Size([2097152, 1])

        # torch.max(signed_distance)
        # tensor(2.8127, device='cuda:0', grad_fn=<MaxBackward1>)
        # torch.min(signed_distance)
        # tensor(-0.6842, device='cuda:0', grad_fn=<MinBackward1>)

        density = self._sdf_to_density(signed_distance) # torch.Size([2097152, 1])

        # torch.max(density)
        # tensor(1.0000, device='cuda:0', grad_fn=<MaxBackward1>)
        # torch.min(density)
        # tensor(1.8543e-25, device='cuda:0', grad_fn=<MinBackward1>)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            ) # torch.Size([2097152, 3])

            # self.sdf.center
            # Parameter containing:
            # tensor([[0., 0., 0.]], device='cuda:0', requires_grad=True)
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas, # torch.Size([2097152, 1])
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1) # torch.Size([2097152, 3])
        }

        return out
    
"""
what is the idea behind using this formula as density
-torch.log(1.0 - density) / deltas

The formula \(-\text{torch.log}(1.0 - \text{density}) / \text{deltas}\) used for the density in the provided code is associated with the rendering of volumes in computer graphics, especially in the context of raymarching and volume ray casting. Let's break it down:

1. **Density Scaling with Deltas**: 
   - Dividing by `deltas` scales the density based on the segment length between two sample points on a ray. If two samples are very close to each other, you expect the change in density to be small and vice versa. This normalization is important to ensure consistent and realistic rendering regardless of the sampling resolution.

2. **Negative Logarithm Transformation**: 
   - The term \(-\text{torch.log}(1.0 - \text{density})\) is a common transformation in volume rendering. The idea behind this is related to the Beer-Lambert law from optics, which describes the attenuation of light as it passes through a medium.
   - The basic intuition is: if you have a very high density (close to 1), then \(1.0 - \text{density}\) is close to 0, and the negative logarithm of a value close to 0 is a large positive number, indicating high attenuation or opacity. Conversely, if the density is low, the negative logarithm produces a value closer to 0, indicating low attenuation or transparency.
   - This transformation is especially useful for raymarching. As you accumulate color and opacity along a ray, this formula ensures that regions of high density contribute more to the final color and opacity, making them appear more "solid".

3. **Why Not Use Density Directly**:
   - Without transformation, using raw density values could lead to unrealistic results. Regions of slightly different densities might appear nearly identical, and subtle features could be lost. The logarithmic transformation accentuates differences in densities, making variations more perceptible and producing a more visually appealing and informative result.

In summary, the given formula effectively scales and transforms the raw density values derived from the SDF into values more suitable for volume rendering. It ensures that the rendered volume has a realistic appearance with discernible features, even if the original density differences were subtle.
"""

"""
what is the idea behind using this formula for feature
'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1) # torch.Size([2097152, 3])

The formula for `feature` in the provided code seems to represent the color or visual attribute of each sampled point within the volume. Let's break down its components:

1. **base_color**: 
   - This variable determines the underlying color of the sample point. If the `rainbow` flag is set, this color changes based on the distance from the center of the SDF, which can be a way to visualize the depth or layers of the volume. Otherwise, it has a fixed value (`1.0`), making it neutral and not affecting the final color.

2. **self.feature**: 
   - This is a learnable parameter from the model, meaning it can be adjusted during training or optimization. It represents a global color modulation factor for the entire volume. By adjusting `self.feature`, the model can learn the optimal color intensity or weighting for the best visual representation or some other objective.

3. **density.new_ones(sample_points.shape[0], 1)**:
   - This creates a tensor of ones with the same datatype and device as `density` and with the shape `(sample_points.shape[0], 1)`. Essentially, it's a tensor filled with the value `1.0` and has the same number of rows as `sample_points`. This tensor acts as a multiplier, ensuring that the feature value remains consistent across all sample points. In this particular formula, it seems to be redundant since multiplying by a tensor of ones won't change the value. However, it's possible that in other contexts or variations of the code, this tensor might be replaced with another factor, and keeping the structure consistent makes it easier to make such changes.

The product of these three components gives the final `feature` value for each sample point. The idea behind this formula is to modulate the basic color (`base_color`) of each point with a global feature scaling (`self.feature`) and ensure it's consistent across all points (multiplied by the tensor of ones).

In essence, this `feature` formula allows both global and local adjustments to the color attributes of the volume. The `base_color` provides local differentiation, `self.feature` offers a global modulation, and the tensor of ones ensures consistent shaping across the dataset.
"""

"""
explain how this function converts sdf values to densities
def _sdf_to_density(self, signed_distance):
    # Convert signed distance to density with alpha, beta parameters
    return torch.where(
        signed_distance > 0,
        0.5 * torch.exp(-signed_distance / self.beta),
        1 - 0.5 * torch.exp(signed_distance / self.beta),
    ) * self.alpha

This function `_sdf_to_density` aims to convert the values of a signed distance function (SDF) to density values. Let's break down its steps and the underlying rationale:

1. **Signed Distance Values**:
   - An SDF gives us the closest distance to a surface, with a positive value meaning outside the surface and a negative value meaning inside. The value `0` typically denotes being exactly on the surface.

2. **Exponential Transformation**:
   - The function applies an exponential transformation to the SDF values. The rationale behind this is to create rapidly decreasing density values as you move farther from the surface.
   - `torch.exp(-signed_distance / self.beta)` ensures that the density drops off exponentially as the `signed_distance` increases. The parameter `self.beta` controls the rate of this drop-off.

3. **Density Calculation**:
   - For `signed_distance > 0` (outside the surface): The density is given by `0.5 * torch.exp(-signed_distance / self.beta)`. As the distance from the surface increases, the density value drops off, approaching 0.
   - For `signed_distance <= 0` (inside the surface): The density is `1 - 0.5 * torch.exp(signed_distance / self.beta)`. The use of the `1 - ...` formula ensures that, inside the object, the density starts off high (close to 1) right beneath the surface and decreases as we move inwards.

4. **Alpha Modulation**:
   - Finally, the densities are multiplied by `self.alpha`. This is a scaling factor that can be seen as a global control over the overall opacity or density of the object. If `self.alpha` is a small value, the object will appear more transparent throughout, while a larger `self.alpha` will make it appear denser.

In essence, this function ensures that:
- Points on or very close to the surface have the highest density.
- The density drops off rapidly as you move away from the surface, both inside and outside the object.
- The overall density can be controlled using the `self.alpha` parameter. 

The conversion from SDF to density in this manner is useful in volume rendering, especially in raymarching, because it helps to visualize surfaces and boundaries within the volume clearly. The rapid exponential drop-off ensures that regions away from the surface do not obscure the surface itself in the rendered image.
"""

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input: # True
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)

# x[..., None].shape
# torch.Size([1024, 3, 1])
# self._frequencies
# tensor([1., 2.], device='cuda:0')
# (x[..., None] * self._frequencies).shape
# torch.Size([1024, 3, 2])
# embed.shape
# torch.Size([1024, 6])
# embed.sin().shape
# torch.Size([1024, 6])
# embed.cos().shape
# torch.Size([1024, 6])
# x.shape
# torch.Size([1024, 3])
# torch.cat((embed.sin(), embed.cos(), x), dim=-1).shape
# torch.Size([1024, 15])

class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        # MLPWithInputSkips                   (n_layers,     input_dim,  output_dim,      skip_dim,             hidden_dim,             input_skips)
        output_dim = 128
        self.MLP = MLPWithInputSkips(cfg.n_layers_xyz, embedding_dim_xyz, output_dim , embedding_dim_xyz, cfg.n_hidden_neurons_xyz, cfg.append_xyz)

        #MLP layers
        self.linear1 = torch.nn.Linear(output_dim, 4)
        self.linear2 = torch.nn.Linear(output_dim, 64)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, ray_bundle):
        # TODO (forwards pass)
        direction_embedding = self.harmonic_embedding_dir(ray_bundle.directions) # torch.Size([1024, 15])
        position_embedding = self.harmonic_embedding_xyz(ray_bundle.sample_points.view(-1, 3)) # torch.Size([131072, 39])

        output = {"density": None, "feature": None}
        out = self.linear1(self.MLP(position_embedding, position_embedding)) # torch.Size([131072, 4])

        output["density"] = self.relu(out[:, 0]) # torch.Size([131072])
        output["feature"] = self.sigmoid(out[:, 1:]) # torch.Size([131072, 3])

        return output

# self.MLP(position_embedding, position_embedding).shape
# torch.Size([131072, 128])

volume_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
}