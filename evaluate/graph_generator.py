import os
import json
import numpy as np
import random
import argparse
import pickle

import torch
import torch_geometric
from torch_geometric.io import read_ply

path = "Simplified_Meshes/"
move_path = "Mesh_Graphs/"

os.mkdir(move_path)
mesh_files = os.listdir(path)
print("mesh files ",mesh_files)
for file in mesh_files:
	
	if(file.endswith(".obj")):
		full_mesh_path = path +  file 
		graph_path = move_path +  file[0:len(file)-4] +".pickle"
		print("graph_path ",graph_path)
		if(os.path.exists(full_mesh_path)):
			print("came here ")
			mesh = read_ply(full_mesh_path);
			pre_transform = torch_geometric.transforms.FaceToEdge();
			graph =pre_transform(mesh);

			with open(graph_path, 'wb') as f:
				pickle.dump(graph, f, protocol=2)
		



