import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

	# loss = torch.nn.BCELoss() # wont work because of bounds issue
	
	# relu_activation = torch.nn.ReLU()
	# loss = torch.nn.BCELoss()
	# output = loss(relu_activation(voxel_src), voxel_tgt)

	# loss = torch.nn.BCEWithLogitsLoss() # works
	# output = loss(voxel_src, voxel_tgt)

	sigmoid_activation = torch.nn.Sigmoid()
	loss = torch.nn.BCELoss()
	output = loss(sigmoid_activation(voxel_src), voxel_tgt)

	# implement some loss for binary voxel grids
	return output

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian