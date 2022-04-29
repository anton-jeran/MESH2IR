from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
# from PIL import Image
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch_geometric
from torch_geometric.io import read_ply
import librosa

import io
import sys

from miscc.config import cfg

#embeddings = [mesh_path,RIR_path,source,receiver]
class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',rirsize=4096): #, transform=None, target_transform=None):

        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.bbox = None
        
  
        self.embeddings = self.load_embedding(data_dir)

    def get_RIR(self, full_RIR_path):
        # wav,fs = sf.read(full_RIR_path) 
        wav,fs = librosa.load(full_RIR_path)
 
        # wav_resample = librosa.resample(wav,16000,fs)
        wav_resample = librosa.resample(wav,orig_sr=fs,target_sr=16000)

        length = wav_resample.size

        crop_length = 3968 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            std_value = np.std(wav_resample) * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new = np.concatenate([wav_resample,zeros])/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])
        else:
            wav_resample_new = wav_resample[0:crop_length]
            std_value = np.std(wav_resample_new)  * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new =wav_resample_new/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])

        resample_length = int(self.rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        return RIR


    # def get_graph(self, full_mesh_path):
    #     mesh = read_ply(full_mesh_path);
    #     pre_transform = torch_geometric.transforms.FaceToEdge();
    #     graph =pre_transform(mesh);
    #     # edge_index = graph['edge_index']
    #     # vertex_position = graph['pos']
        
    #     return graph #edge_index, vertex_position


    def get_graph(self, full_graph_path):
        
        with open(full_graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        return graph #edge_index, vertex_position

    def load_embedding(self, data_dir):
        embedding_filename   = '../embeddings.pickle'  
        with open(embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings


    def __getitem__(self, index):
      

        graph_path,RIR_path,source_location,receiver_location = self.embeddings[index]

        data_dir = self.data_dir

        full_graph_path = os.path.join(data_dir,graph_path)
        full_RIR_path  = os.path.join(data_dir,RIR_path)
        source_receiver = source_location+receiver_location
        embedding = np.array(source_receiver).astype('float32')
        RIR = self.get_RIR(full_RIR_path)

        graph = self.get_graph(full_graph_path);
        graph.RIR = RIR
        graph.embeddings = embedding

      

        # print("shape ", transpose_edge_index.shape)
        return graph
        
    def __len__(self):
        return len(self.embeddings)
