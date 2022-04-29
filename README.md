# MESH2IR: Neural Acoustic Impulse Response Generator for Complex 3D Scenes

This is the official implementation of our mesh-based neural network (MESH2IR) to generate acoustic impulse responses (IRs) for indoor 3D scenes represented
using a mesh.

## Requirements

```
Python3.6
pip3 install numpy
pip3 install torch
pip3 install torchvision
pip3 install python-dateutil
pip3 install soundfile
pip3 install pandas
pip3 install scipy
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip3 install librosa
pip3 install easydict
pip3 install cupy-cuda102
pip3 install wavefile
pip3 install torchfile
pip3 install pyyaml==5.4.1
pip3 install pymeshlab
pip install openmesh
pip3 install gdown

```
## Download Data
Please follow the instructions and sign the agreements in this [**link**](https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html?msclkid=f7bd359dc76411eca640dbcac3538f68) before downloading any files related to [**3D-FRONT datset**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).  
## Generating RIRs using the trained model

Download the trained model, sample 3D indoor envrionemnt meshes from [**3D-FRONT dataset**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), and sample source receiver paths files.

```
source download_files.sh

```

Simplify the sample 3D indoor environment meshes.

```
python3 mesh_simplification.py

```

Convert simplified meshes to graph.

```
python3 graph_generator.py

```



