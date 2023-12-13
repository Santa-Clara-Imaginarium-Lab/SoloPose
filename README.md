# SoloPose
SoloPose is a novel one-shot, many-to-many spatio-temporal transformer model for kinematic 3D human pose estimation of video.  SoloPose is further fortified by HeatPose, a 3D heatmap based on Gaussian Mixture Model distributions that factors target key points as well as kinematically adjacent key points.
<div align="center">
    <img src="assest/human3.6M_cooridnates_errors.png", width="900">
</div>

<div align="center">
    <img src="assest/heatMap.png", width="900">
</div>


# News!
- Dec 2023: [SoloPose](https://github.com/Santa-Clara-Media-Lab/SoloPose) is released!
- Nov 2023: [MoEmo](https://github.com/Santa-Clara-Media-Lab/MoEmo_Vision_Transformer) codes are released!
- Oct 2023: MoEmo was accepted by IROS 2023 (IEEE/RSJ International Conference on Intelligent Robots and Systems).

# install

we need to install the [huggingface](https://huggingface.co/docs/transformers/installation#:~:text=%F0%9F%A4%97%20Transformers%20is%20tested%20on,PyTorch%20installation%20instructions.) to apply the pre-train model. Besides, our network is implemented by Pytorch.


```
conda create -n pose python=3.8
conda activate pose
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python
pip install 'transformers[torch]'
pip install matplotlib
pip install moviepy
pip install chardet
pip install scipy
git clone https://github.com/Santa-Clara-Media-Lab/SoloPose.git
cd Xpose
```
## visuial

- FFmpeg (if you want to export MP4 videos)
- ImageMagick (if you want to export GIFs)

# Pre-trained model
Our pre-trained models will be released after our paper has been accepted.

# Data Prepare
Please ensure you have done everything before you move to the following steps.
