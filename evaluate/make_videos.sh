#!/bin/bash

ffmpeg -y -i Video/embeddings_0/embeddings_0.mp4 -i Output/embeddings_0/output_MESH2IR.wav -map 0:v -map 1:a -c:v copy -shortest Output/embeddings_0/embeddings_0-MESH2IR.mp4
ffmpeg -y -i Video/embeddings_1/embeddings_1.mp4 -i Output/embeddings_1/output_MESH2IR.wav -map 0:v -map 1:a -c:v copy -shortest Output/embeddings_1/embeddings_1-MESH2IR.mp4
ffmpeg -y -i Video/embeddings_2/embeddings_2.mp4 -i Output/embeddings_2/output_MESH2IR.wav -map 0:v -map 1:a -c:v copy -shortest Output/embeddings_2/embeddings_2-MESH2IR.mp4
