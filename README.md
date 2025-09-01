# OmniStereo

This repository contains python implementation for paper "OmniStereo: Real-time Omnidireactional Depth Estimation with Multiview Fisheye Cameras".



## Requirements

GCC (tested on GCC 7.5.0)

Python (tested on Python 3.8)

- PyTorch (tested on  PyTorch 1.12.1 with CUDA 11.3)

- TensorBoard

- OpenCV

- SciPy

- tqdm

- [ocamcalib](https://github.com/matsuren/ocamcalib_undistort)



## Dataset

Download `Urban`(cloudy, sunny and sunset), `OmniHouse` and `OmniThings` from [Omnidirectional Stereo Dataset](https://github.com/hyu-cvlab/omnimvs-pytorch).

Note: The pose information in `poses.txt` is different from that in `config.yaml`. Please refer to `config.yaml` as the standard.



## Setup

Compile the interpolation operator.

```bash
cd utils/iter_pixels_cuda
python setup.py install
cd ../..
```



## Evaluation

Evaluate on `Urban` dataset.

```bash
python test.py --data-path [path to dataset] --ocam-path [path to ocams and poses] --val-list ./dataloader/data_list/urban_val.txt --loadmodel ./checkpoints/finetune/epoch29.tar --crop
```



Evaluate on `OmniHouse` dataset.

```bash
python test.py --data-path [path to dataset] --ocam-path [path to ocams and poses] --val-list ./dataloader/data_list/omnihouse_val.txt --loadmodel ./checkpoints/finetune/epoch29.tar
```



Evaluate on `OmniThings` dataset.

```bash
python test.py --data-path [path to dataset] --ocam-path [path to ocams and poses] --val-list ./dataloader/data_list/omnithings_val.txt --loadmodel ./checkpoints/finetune/epoch29.tar
```



## Acknowledgements

The partial code in the `models/spherical_sweep.py` file is adapted from [OmniMVS](https://github.com/matsuren/omnimvs_pytorch).

The partial code in the `models/stereo_matching.py` and `models/submodule.py` files is adapted from [Fast-ACVNet](https://github.com/gangweiX/Fast-ACVNet).

Thanks for opening source of their excellent works.


