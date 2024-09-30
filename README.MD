## INSTALL ENV in Jetson from scratch
```
sudo apt update
sudo apt upgrade
sudo dpkg --configure -a
sudo chmod 777 -R /home/*
alias python=python3
alias pip=pip3

sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

python -m pip install --upgrade pip
pip3 install torch torchvision

sudo apt-get update && apt-get install -y --no-install-recommends \
   build-essential \
   cmake \
   libzmqpp-dev \
   libopencv-dev 

python install ncnn --verbose
pip install 'https://github.com/jetson-nano-wheels/python3.6-pycuda-2021.1/releases/download/v0.0.1/pycuda-2021.1-cp36-cp36m-linux_aarch64.whl'
sudo apt install libvulkan1
sudo apt install libvulkan-dev

PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda/bin:$PATH
sudo -H pip install --no-cache-dir -U jetson-stats
sudo systemctl set-default multi-user.target

pip uninstall scipy
sudo apt-get install python3-scipy
```

## How to run program

```python
cd code/
python3 run.py
```

## How to export tensorrt from onnx
/usr/src/tensorrt/bin/trtexec  --onnx=slowonly.onnx --saveEngine=slowonly.trt --fp16


## Link weights && input test
[weights](https://drive.google.com/drive/folders/1Vs_rqTVe6uewF_XtOIbanNreDgghewM8?usp=sharing)
[inp](https://drive.google.com/drive/folders/1B31wtSz10EdlS4htH7vnZD91NR4xOhQ3?usp=sharing)


## Custom lite weight yolov8 => boost to speed 
[source](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)

```
!gdown --id 1mv7uIBmmtoZ1t_DN3XuPhyoJ84aG0j-X
!gdown --id 1xFNc5Gi5SBKIFxCsTYVqpUEW14SP3-1s
!cp -r head.py /usr/local/lib/python3.10/dist-packages/ultralytics/nn/modules/head.py
!cp -r block.py /usr/local/lib/python3.10/dist-packages/ultralytics/nn/modules/block.py
```

```python
from ultralytics import YOLO
# Create a model
model = YOLO('v8_n.pt')
model.export(format='onnx', half=True, imgsz=640)
```

## Convert slow_only to onnx
[link](https://drive.google.com/drive/folders/1h5ZShSWUM_NxYBuqS77-4pzx76WNHb3S?usp=sharing)
