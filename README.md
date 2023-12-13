# SoloPose
SoloPose is a novel one-shot, many-to-many spatio-temporal transformer model for kinematic 3D human pose estimation of video.

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
git clone https://github.com/stm233/3D_Pose_Estimation.git Xpose
cd Xpose
```
