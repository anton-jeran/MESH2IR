import IPython.display as ipd
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy import signal
import os
import json
import numpy as np
from tqdm import tqdm
import re
import soundfile as sf
import librosa
from pydub import AudioSegment
from scipy.spatial import KDTree

# Define macros
SAMPLE_RATE = 16000

def sample_path(path, delta):
    idx = [0]
    prev_xyz = path[0, 1:4]
    dist = 0
    for i, p in enumerate(path):
        xyz = p[1:4]
        dist = dist + np.linalg.norm(xyz - prev_xyz)
        if dist >= delta:
            dist = 0
            idx.append(i)
        prev_xyz = xyz
    if dist > 0:
        idx.append(len(path)-1)
    return idx

def load_cam_path(cam_path):
    data = np.loadtxt(cam_path, delimiter=',', skiprows=1)
    src = data[data[:, 0] < 0]
    lis = data[data[:, 0] >= 0]
    return src, lis

def time_to_frame(time, rate):
    if np.isscalar(time):
        return int(time * rate)
    else:
        return np.array(time * rate).astype(int)        

def parse_config(sim_config):
    # Parse GWA config
    with open(sim_config, 'r') as f:
        config = json.load(f)
    nsrc = len(config['sources'])
    nrec = len(config['receivers'])
    src_pos = np.zeros((nsrc, 3), dtype=float)
    rec_pos = np.zeros((nrec, 3), dtype=float)
    for p in config['sources']:
        src_pos[int(p['name'][1:])-1] = p['xyz']
    for p in config['receivers']:
        rec_pos[int(p['name'][1:])-1] = p['xyz']
    return src_pos, rec_pos

def normalize_audio(signal, limit=0.9):
    gain = limit / max(max(signal), -min(signal))
    return signal * gain    

def crossfade_audio(seg1, seg2, crossfade):
    ind = np.arange(crossfade)
    w1 = np.cos(ind / (crossfade-1) * np.pi / 2) ** 2
    w2 = 1 - w1
    xf = seg1[-crossfade:] * w1 + seg2[:crossfade] * w2
    combined = np.concatenate((seg1[:-crossfade], xf, seg2[crossfade:]))
    return combined


def auralize_path(config_path, mode, data_folder=None, full_config=None, path_delta=0.1):
    with open(config_path, 'r') as f:
        config = json.load(f)
    data_dir = data_folder #os.path.dirname(config_path)    
    cam_path = os.path.join(os.path.dirname(config_path), config['camera_path'])
    src, lis = load_cam_path(cam_path)
    key_idx = sample_path(lis, path_delta)
    
    # for each segment, find corresponding time range in the dry sound, and pair it with an IR
    upper_time = max(lis[:, 0])
    rendered_sound = dict()
    suffix = mode
    
    # for debugging
    ir_energies = dict()
    seg_energies = dict()
    used_IRs = []

    if mode == "GWA":
        assert data_folder is not None
        _, rec = parse_config(full_config)
        rec_tree = KDTree(rec)

    for source in config['sources']:    
        ir_energies[source['name']] = []
        seg_energies[source['name']] = []

        audio_path = source['audio']  # assumes absolute path
        try:
            src_volume = float(source['volume'])
        except KeyError:
            src_volume = 1.0
        dry_sound, fs = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(dry_sound) < time_to_frame(upper_time, SAMPLE_RATE):
            n_fold = time_to_frame(upper_time, SAMPLE_RATE) // len(dry_sound) + 1
            dry_sound = np.tile(dry_sound, n_fold)

        
        irs = []
        for idx in key_idx[:-1]:
            if mode == "MESH2IR":
                ir_folder = os.path.join(data_dir, source['name'])
                ir, _ = librosa.load(f'{ir_folder}/{idx+1}.wav', sr=SAMPLE_RATE)
            elif mode == "GWA":
                kth = 1
                while True:                
                    closest_idx = rec_tree.query(lis[idx, 1:4], [kth])[1][0]
                    ir_path = os.path.join(data_folder, f'L{source["name"][1:]}_R{closest_idx+1:04}.wav')
                    if os.path.exists(ir_path):
                        break                
                    kth += 1
                    print(f'{closest_idx} failed, try next neighbor rank {kth}')
                ir, _ = librosa.load(ir_path, sr=SAMPLE_RATE)
                used_IRs.append(ir_path)
            else:
                raise Exception("Undefined data source!")
            if len(ir.shape) > 1:
                ir = ir[:, 0]
            irs.append(ir)

        maxlen = max([len(ir) for ir in irs])  # unify ir lengths
        for i in range(len(irs)):        
            if len(irs[i]) < maxlen:
                irs[i] = np.pad(irs[i], (0, maxlen - len(irs[i])), mode='constant')
        irs = np.array(irs)
        irlen = irs.shape[1]

        output_wav = []
        for i in range(len(key_idx)-1):
            start_frame = time_to_frame(lis[key_idx[i], 0], SAMPLE_RATE)
            end_frame = time_to_frame(lis[key_idx[i+1], 0], SAMPLE_RATE) + irlen
            dry_segment = dry_sound[start_frame:end_frame]
            segment = scipy.signal.convolve(dry_segment, irs[i])
            ir_energies[source['name']].append(sum(irs[i] * irs[i])/len(irs[i]))
            seg_energies[source['name']].append(sum(segment * segment)/len(segment))
            if i == 0:
                output_wav = segment
            else:
                output_wav = crossfade_audio(output_wav[:-irlen], segment, irlen)

        output_wav = np.array(output_wav)
        rendered_sound[source['name']] = output_wav * src_volume    
        
    rendered_sound['all'] = np.zeros_like(output_wav)
    for name, audio in rendered_sound.items():
        rendered_sound['all'] += audio
    save_path = f'{data_dir}/output_{suffix}.wav'
    output_all = normalize_audio(rendered_sound['all'])
    print(f'file written to {save_path}')
    sf.write(save_path, output_all, SAMPLE_RATE)
    
    return rendered_sound



targets = ['embeddings_0','embeddings_1','embeddings_2']

for f in targets:
    config_path = f'Video/{f}/aural_config.json'
    full_config = f'/scratch/anton/SIGGRAPH/3D-FRONT-Auralization/aural_sim_results/hybrid/{f}/sim_config.json'    
    data_folder = f'Output/{f}/'
    # rendered_gwa = auralize_path(config_path, 'GWA', data_folder=data_folder, full_config=full_config)    
    rendered_mesh2ir = auralize_path(config_path, 'MESH2IR',data_folder)