import os
import json
import numpy as np
import random
import argparse
import pickle
import math


embedding_list=[]
os.mkdir("Embeddings")
path = "Paths/"
mesh_files = os.listdir(path)

embed_c = 0

for file in mesh_files:
	embedding_list=[]

	mesh_path = file[0:len(file)-8] + ".pickle"

	filepath = path + file
	data=np.loadtxt(filepath, delimiter=',')

	len_data = len(data)


	source_list = []
	listener_list = []
	for j in range(len_data):
		if(data[j][0]<0):
			source_list.append(data[j][1:].tolist())
		else:
			listener_list.append(data[j][1:].tolist())
	
	num_listeners = len(listener_list)
	num_sources = len(source_list)
	print("num sources  ",num_sources,"   num listeners   ",num_listeners)
	for s in range(num_sources):
		for l in range(num_listeners):
			folder_name = "S"+ str(s+1)
			wave_name = str(l+1) +".wav"
			embeddings = [mesh_path,folder_name,wave_name,source_list[s],listener_list[l]]
	
			embedding_list.append(embeddings)






	print("embdiing_list", len(embedding_list))
	filler = 256  - (len(embedding_list) % 256)
	len_embed_list = len(embedding_list) -1
	if(filler < 256):
		for i in range(filler):
			embedding_list.append(embedding_list[len_embed_list-filler+i])
	print("embdiing_list123", len(embedding_list))


	embeddings_pickle ="Embeddings/embeddings_"+str(embed_c)+".pickle"
	embed_c = embed_c + 1
	with open(embeddings_pickle, 'wb') as f:
		pickle.dump(embedding_list, f, protocol=2)




