
# Prepare environment

```shell
conda create -n t2v python=3.8
conda activate t2v
pip3 install -r requirements/pt13.txt
pip3 install .

##nuscenes related pakages
sudo pip install mmcv-full==1.6.0
sudo pip install mmdet==2.28.2
sudo pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
sudo pip install -e .
```

**Note:**

Some pakages may still be missing:

```shell
##flash-attention:
export CUDA_HOME=/usr/local/cuda-11.7 ##specify your own cuda path
pip install flash-attn --no-build-isolation

pip install ipython
pip install boto3
pip install cffi
pip install influxdb
```
Some version of pakages may be incompatiable：

```shell
pip uninstall numpy
pip install numpy==1.23.3
pip install -U numba
pip uninstall open-clip-torch
pip install open-clip-torch==2.20.0

```

If you have further issues about the environment, you can check with the [**My Panacea Env.**](./panacea.yml) for the detailed comparison.

