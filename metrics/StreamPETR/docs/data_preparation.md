# Data Preparation

## Real Dataset
**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**2. Infos file**

[train](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_train.pkl), [val](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_val.pkl) and [test](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_test.pkl) pkl.


After preparation, you will be able to see the following directory structure:  

**Folder structure**
```
StreamPETR
├── projects/
├── mmdetection3d/
├── tools/
├── configs/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes2d_temporal_infos_train.pkl
|   |   ├── nuscenes2d_temporal_infos_val.pkl
```
## Gen-nuScenes Dataset
**1. Download Gen-nuScenes**

Download the [Gen-nuScenes training dataset](https://huggingface.co/datasets/orangewen/Gen-nuScenes/resolve/main/gen-nuscenes-train.tar.gz?download=true) to `./data/nuscenes`.

Download the [Gen-nuScenes validation dataset](https://huggingface.co/datasets/orangewen/Gen-nuScenes/resolve/main/gen-nuscenes-val.tar.gz?download=true) to `./data/nuscenes`.

Download the training pkl file [Gen-nuScenes-train-infos.pkl](https://huggingface.co/datasets/orangewen/Gen-nuScenes/resolve/main/gen-nuscenes-train-infos.pkl?download=true) to `./data/nuscenes`.

Download the validation pkl file [Gen-nuScenes-val-infos.pkl](https://huggingface.co/datasets/orangewen/Gen-nuScenes/resolve/main/gen-nuscenes-val-infos.pkl?download=true) to `./data/nuscenes`.

**2. unzip Gen-nuScenes**
```shell
tar -zxvf **.tar.gz
ln -s /data/code/StreamPETR/data/nuscenes/data/Dataset/nuScenes/gen-nuscenes-val/ gen-nuscenes-val
ln -s /data/code/StreamPETR/data/nuscenes/data/Dataset/nuScenes/gen-nuscenes-train/ gen-nuscenes-train
```