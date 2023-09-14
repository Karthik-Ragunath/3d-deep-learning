import torch
import pytorch3d
import numpy as np
import os
from PIL import Image

from pytorch3d.renderer import OpenGLPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh import TexturesUV, TexturesVertex
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import HardPhongShader
from pytorch3d.io import save_obj
from pytorch3d.io import IO
import matplotlib.pyplot as plt

# Define the vertices and faces of the tetrahedron (as shown in the previous answers)
vertices = torch.tensor([
    [1.0, 1.0, 1.0],  # Vertex 0
    [1.0, -1.0, -1.0],  # Vertex 1
    [-1.0, 1.0, -1.0],  # Vertex 2
    [-1.0, -1.0, 1.0],  # Vertex 3
], dtype=torch.float32)

faces = torch.tensor([
    [0, 1, 2],  # Face 0 (vertices 0, 1, 2)
    [0, 1, 3],  # Face 1 (vertices 0, 1, 3)
    [0, 2, 3],  # Face 2 (vertices 0, 2, 3)
    [1, 2, 3],  # Face 3 (vertices 1, 2, 3)
], dtype=torch.int64)

# Generate UV coordinates for the tetrahedron
uv_coords = torch.tensor([
    [0.0, 0.0],  # Vertex 0 UV
    [1.0, 0.0],  # Vertex 1 UV
    [1.0, 1.0],  # Vertex 2 UV
    [0.0, 1.0],  # Vertex 3 UV
], dtype=torch.float32)

# Create a TexturesUV object from the UV coordinates
# textures = TexturesUV(uvs=[uv_coords])
texture_image = torch.ones((1, vertices.shape[0], 3), dtype=torch.float32).to(vertices.device)
texture = TexturesVertex(verts_features=texture_image)

# Create a Meshes object from the vertices and faces
mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces], textures=texture)

# Create a camera and renderer
R, T = look_at_view_transform(2.7, 90, 60, device=vertices.device)
cameras = OpenGLPerspectiveCameras(device=vertices.device, R=R, T=T)
renderer = pytorch3d.renderer.MeshRenderer(
    rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras),
    shader=pytorch3d.renderer.HardPhongShader(device=vertices.device, lights=None),
)

# Place a point light in front of the cow.
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=vertices.device)

# Render the mesh with UV texture
images = renderer(mesh, cameras=cameras, lights=lights)

# Define output directory
output_dir = "starter/output_uv_map"
os.makedirs(output_dir, exist_ok=True)

# Save the mesh and texture as .obj, .mtl, and .img files
# obj_filename = os.path.join(output_dir, "tetrahedron.obj")
# mtl_filename = os.path.join(output_dir, "tetrahedron.mtl")
# img_filename = os.path.join(output_dir, "tetrahedron.png")

# # Save the mesh as .obj file
# save_obj(
#     obj_filename,
#     verts=vertices,
#     faces=faces,
#     textures=texture,
#     include_textures=True,
#     mtl_filename=mtl_filename,
# )

obj_filename = os.path.join(output_dir, "tetrahedron.ply")
img_filename = os.path.join(output_dir, "tetrahedron.png")
IO().save_mesh(
    mesh,
    obj_filename,
    binary=False,
    include_textures=True
)

plt.imsave(img_filename, images[0, ..., :3].cpu().numpy())

# # Save the texture image as .img file
# texture_image = (images[0, ..., :3] * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
# texture_image = Image.fromarray(texture_image)
# texture_image.save(img_filename)

print(f"Mesh saved to: {obj_filename}")
# print(f"Material file saved to: {mtl_filename}")
print(f"Texture image saved to: {img_filename}")
