
## Introduction

This repository is for Panancea's evaluation, based on the implementation of StreamPETR [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.11926).

## Getting Started

Please follow our documentation step by step.

1. [**Environment Setup.**](./docs/setup.md)
2. [**Data Preparation.**](./docs/data_preparation.md)

## Evaluation on Gen-nuScenes

###
Download the [pretrained baseline model](https://huggingface.co/datasets/orangewen/Gen-nuScenes/resolve/main/iter_42192.pth?download=true) to `./work_dirs/streampetr_r50_atss_512x256_nopretrain/`.

### Comparison of the generated data with real data on the validation set.
（1）test on real nuScenes validation set using pretrained baseline model StreamPETR.
```bash
tools/dist_test.sh projects/configs/diffusion/streampetr_r50_atss_512x256_withpretrain_real_finetune.py work_dirs/streampetr_r50_atss_512x256_nopretrain/iter_42192.pth 8 --eval bbox
```
（2）test on Gen-nuScenes validation set using pretrained baseline model StreamPETR.
```bash
tools/dist_test.sh projects/configs/diffusion/streampetr_r50_atss_512x256_window.py work_dirs/streampetr_r50_atss_512x256_nopretrain/iter_42192.pth 8 --eval bbox
```

### Comparison involving data augmentation using synthetic data.
（1）train on real nuScenes training set (baseline).
```bash
tools/dist_train.sh projects/configs/diffusion/streampetr_r50_atss_512x256_nopretrain.py 8 --work-dir work_dirs/streampetr_r50_atss_512x256_nopretrain/
```
（2）pretrain on Gen-nuScenes training set.
```bash
tools/dist_train.sh projects/configs/diffusion/streampetr_r50_atss_512x256_window_pseudo_pretrain.py 8 --work-dir work_dirs/streampetr_r50_atss_512x256_window_pseudo_pretrain/
```
（3）finetune on real nuScenes training set.
```bash
tools/dist_train.sh projects/configs/diffusion/streampetr_r50_atss_512x256_withpretrain_real_finetune.py 8 --work-dir work_dirs/streampetr_r50_atss_512x256_withpretrain_real_finetune/
```
* Remember to change the resume path to your pretrained model.
* After finetuning on real nuScenes training set, the automatically test may encounter errors. You can manually run the test command to test the latest checkpoint using the following command:
```bash
tools/dist_test.sh projects/configs/diffusion/streampetr_r50_atss_512x256_withpretrain_real_finetune.py work_dirs/streampetr_r50_atss_512x256_withpretrain_real_finetune/iter_42192.pth 8 --eval bbox
```

## Results

### Comparison of the generated data with real data on the validation set.
| Data | Image Size |NDS|
| :---: | :---: | :---: |
|Real| 512 * 256| 46.9 |
|Gen-nuScenes| 512 * 256| 32.1 (68%)|

### Comparison involving data augmentation using synthetic data.
| Real | Generated | mAP↑| mAOE↓| mAVE↓| NDS↑|
| :---: | :---: | :---: | :---: | :---: |:---: |
|✓| - | 34.5 |59.4 |29.1 |46.9|
|- |✓ |22.5| 72.7| 46.9| 36.1|
|✓ |✓ |37.1(+2.6%)| 54.2| 27.3| 49.2 (+2.3%)|


## Citation
If you find Panacea and StreamPETR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@article{wang2023exploring,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2303.11926},
  year={2023}
}

@article{wen2023panacea,
  title={Panacea: Panoramic and Controllable Video Generation for Autonomous Driving},
  author={Wen, Yuqing and Zhao, Yucheng and Liu, Yingfei and Jia, Fan and Wang, Yanhui and Luo, Chong and Zhang, Chi and Wang, Tiancai and Sun, Xiaoyan and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2311.16813},
  year={2023}
}