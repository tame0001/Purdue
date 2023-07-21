# ECE570 Final Project

GitHub https://github.com/tianzhi0549/FCOS

## FCOS requirement for compiling from source code

python3.8 + torch 1.8.1 because it is the oldest verion that support cuda 11.3.
However, this combination does not support cpu

### Some libs are depreciated. Need to manually change

chanage #include "cpu/vision.h" to #include "vision.h"
from this two files
nano fcos_core/csrc/cpu/nms_cpu.cpp
nano fcos_core/csrc/ROIAlign.h

python3.8 + torch 1.9.0 because it is the oldest verion that support cuda 11.3.
This combination does not support cpu.
The result also failed as previous combination

torch 1.10 or later support gpu but FCOS does not work (build fail)

## Impliment from PyTorch Vision

The data set link:
Training: LSC as of CVPPP2017:

- Training images https://fz-juelich.sciebo.de/s/oGWpbXTbyb9dI52
- Training truth: https://fz-juelich.sciebo.de/s/XKydUI2eSRCVJQ4

Testing: LSC as of CVPPP2017

- Testing images: https://fz-juelich.sciebo.de/s/pm2MvVNljmBxfsr

There are 4 notebooks that I developed

- coco_encoder: This notebook is for creating COCO format data set that will be used to re-train the data
- fcos_before: This notebook is to get result from pre-train model
- fcos_training: This notebook is for training the model which 1 sub data set
- fcos_all_data: This notebook is for training the model with entire data set
