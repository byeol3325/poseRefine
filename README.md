### Installation
This repo is tested on our local environment (python=3.6, cuda=9.0, pytorch=1.1), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n poseRefine python=3.6
```
Then, activate the environment:
```bash
conda activate poseRefine
```

Install  Install PyTorch:

```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt
```

### Data Prepartaion
use makeDataform.py and kitti dataset to prepare dataset
```
# data preparation
|dataset/
  |kitti/
    |train/
      |calibration/
      |poses/
      |rgb/
    |test/
      |calibration/
      |poses/
      |rgb/
```

### train

### eval

