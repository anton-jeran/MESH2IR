import os
import json
import numpy as np
import random
import argparse
import pickle
import math
#embeddings = [mesh_path,RIR_path,source,receiver]

embedding_list=[]

path = "dataset/"
mesh_folders = os.listdir(path)
# num_counter = 9
# temp_counter = 0 
# print("len folders ",len(mesh_folders))


for folder in mesh_folders:
	# mesh_path = folder +"/" + folder +".obj"
	mesh_path = folder +"/" + folder +".pickle"
	RIR_folder  = path + folder +"/hybrid"

	if(os.path.exists(RIR_folder)):
		json_path = RIR_folder +"/sim_config.json"
		json_file = open(json_path)
		data = json.load(json_file)
		# receivers = len(data['receivers'])

		# if(receivers<(num_counter+temp_counter)):
		# 	num_receivers =receivers #len(data['receivers'])
		# 	temp_counter = temp_counter + (num_counter - receivers)
		# else:
		# 	num_receivers = num_counter+temp_counter
		# 	temp_counter = 0

		num_receivers = len(data['receivers'])
		num_sources = len(data['sources'])

		# print("num_receivers  ", num_receivers,"   num_sources  ", num_sources)
		for n in range(num_receivers):
			for s in range(num_sources):
				source = data['sources'][s]['xyz']
				receiver = data['receivers'][n]['xyz']
				RIR_name = "L"+str(data['sources'][s]['name'][1:]) + "_R"  + str(data['receivers'][n]['name'][1:]).zfill(4)+".wav"
				RIR_path = folder +"/hybrid/" + RIR_name
				full_RIR_path = path+ RIR_path
				if(os.path.exists(full_RIR_path)):
					embeddings = [mesh_path,RIR_path,source,receiver]
					embedding_list.append(embeddings)

print("embdiing_list", len(embedding_list))
filler = 128  - (len(embedding_list) % 128)
len_embed_list = len(embedding_list) -1
if(filler < 128):
	for i in range(filler):
		embedding_list.append(embedding_list[len_embed_list-filler+i])

# embed_count = 128*2
# embedding_list = embedding_list[0:embed_count]
# print("embdiing_list12345", len(embedding_list))

embeddings_pickle ="embeddings.pickle"
with open(embeddings_pickle, 'wb') as f:
	pickle.dump(embedding_list, f, protocol=2)




