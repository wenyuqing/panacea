# Environment Setup

## Base Environments
Python >= 3.8 \
CUDA == 11.1 \
PyTorch == 1.9.0 \
mmdet3d == 1.0.0rc6 

**Notes**: 
Please try to install all the packages the same version with the requirements.
E.g. if your numpy by default is higher than 1.23.3, you can pip uninstall it and reinstall the required version.

## Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n streampetr python=3.8 -y
conda activate streampetr
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```
**c. Clone StreamPETR.**
```
git clone https://github.com/wenyuqing/panacea.git

cd metrics/StreamPETR
```

**d. Install mmdet3d.**
```shell
pip install openmim
mim install mmcv-full==1.6.0
pip install mmdet==2.24.1
pip install mmsegmentation==0.26.0
pip install ipython einops fvcore
pip install fairscale==0.4.13
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
# following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation
cd ..
```
**e. Check the version of numpy and yapf.**

If your version of numpy is newer than 1.23.3 or yapf is newer than 0.40.1, some errors may occur.
You can reinstall them:

```shell
pip uninstall numpy
pip uninstall yapf
pip uninstall setuptools
pip install numpy==1.23.3 
pip install yapf==0.40.1 
pip install setuptools==59.5.0
```

**d. Make some modifications to the nuscenes package.**

As the number of samples of Gen-nuScenens may differ from the original real nuScenes, some modifications are needed. \
Please add the following lines to the file : \
/data/miniconda3/envs/streampetr/lib/python3.8/site-packages/nuscenes/eval/detection/evaluate.py
```python
all_annotations = EvalBoxes()
if len(self.pred_boxes.sample_tokens) != len(self.gt_boxes.sample_tokens):
    for key, value in self.gt_boxes.boxes.items():
        if key in self.pred_boxes.sample_tokens:
            all_annotations.add_boxes(key, value)
    self.gt_boxes = all_annotations
```
The insertion location should be the 83th line, 
between
```python
self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
##insert here
assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
```
       