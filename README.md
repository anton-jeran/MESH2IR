# MESH2IR: Neural Acoustic Impulse Response Generator for Complex 3D Scenes

This is the official implementation of our mesh-based neural network ([**MESH2IR**](https://arxiv.org/pdf/2205.09248.pdf)) to generate acoustic impulse responses (IRs) for indoor 3D scenes represented
using a mesh. Our Neural Sound Rendering results is available [**here**](https://anton-jeran.github.io/M2IR/).

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
Please follow the instructions and sign the agreements in this [**link**](https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html?msclkid=f7bd359dc76411eca640dbcac3538f68) before downloading any files related to [**3D-FRONT dataset**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).  


## Generating IRs using the trained model

Codes are available inside **evaluate** folder.

Download the trained model, sample 3D indoor envrionemnt meshes from [**3D-FRONT dataset**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), and sample source receiver paths files. Note that we show a demo with only 5 different meshes. You can download more than 6000 3D indoor scene meshes using the following [**link**](https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html?msclkid=f7bd359dc76411eca640dbcac3538f68). 

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

Generate embedding with different receiver and source locations for a three different 3D indoor scenes. For 3 different indoor scenes, we have stored sample source-recevier locations in a csv format inside **Paths** folder. Columns 2-4 give the 3D cartesian coordinates of the source and receiver positions. Column 1 with negative values corresponds to source positions and Column 1 with non-negative values corresponds to listener positions. 

```
python3 embed_generator.py
```

Generate IRs corresponds to each embedding files inside **Embeddings** folder using the following command.

```
python3 evaluate.py
```

You can find generated IRs inside the **Output** folder.


## Train MESH2IR

Codes are available inside **train** folder.

Download the [**GWA dataset**](https://gamma.umd.edu/researchdirections/sound/gwa) for 100 different meshes using the following command. Note that this is a subset of data that is used to train MESH2IR. You can get the full dataset using the following [**link**](https://gamma.umd.edu/researchdirections/sound/gwa). You need to get 3D-FRONT license before downloading using this [**link**](https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html?msclkid=f7bd359dc76411eca640dbcac3538f68).

```
source download_data.sh
```

Generate embedding with mesh paths, IR paths and source-receiver locations using the following command

```
python3 embed_generator.py
```

To train **MESH2IR**, go inside **MESH2IR** folder and run the following command

```
python3 main.py --cfg cfg/RIR_s1.yml --gpu 0,1
```


To train **MESH2IR-D-EDR**, go inside **MESH2IR-D-EDR** folder and run the following command

```
python3 main.py --cfg cfg/RIR_s1.yml --gpu 0,1
```

Trained models will be available inside **train/output/** folder

## Citations

If you use our **MESH2IR** for your research, please consider citing

```
@article{ratnarajah2021fast,
  title={FAST-RIR: Fast neural diffuse room impulse response generator},
  author={Ratnarajah, Anton and Zhang, Shi-Xiong and Yu, Meng and Tang, Zhenyu and Manocha, Dinesh and Yu, Dong},
  journal={arXiv preprint arXiv:2110.04057},
  year={2021}
}
```

Our work is inspired by [**FAST-RIR**](https://arxiv.org/pdf/2110.04057.pdf).

```
@article{ratnarajah2022mesh2ir,
  title={MESH2IR: Neural Acoustic Impulse Response Generator for Complex 3D Scenes},
  author={Anton Ratnarajah and Zhenyu Tang and Rohith Chandrashekar Aralikatti and Dinesh Manocha},
  journal={arXiv preprint arXiv:2205.09248},
  year={2022}
}
```

If you use 3D indoor scenes from [**3D-FRONT dataset**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), please cite

```
 @inproceedings{fu20213d,
      title={3d-front: 3d furnished rooms with layouts and semantics},
      author={Fu, Huan and Cai, Bowen and Gao, Lin and Zhang, Ling-Xiao and Wang, Jiaming and Li, Cao and Zeng, Qixun and Sun, Chengyue and Jia, Rongfei and Zhao, Binqiang and others},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={10933--10942},
      year={2021}
    }
```

If you use [**GWA dataset**](https://gamma.umd.edu/researchdirections/sound/gwa) to train our model, please cite

```
@misc{tang2022gwa,
      title={GWA: A Large Geometric-Wave Acoustic Dataset for Audio Deep Learning}, 
      author={Zhenyu Tang and Rohith Aralikatti and Anton Ratnarajah and and Dinesh Manocha},
      year={2022},
      eprint={2204.01787},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
