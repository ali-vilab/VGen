# VGen


![figure1](source/VGen.jpg "figure1")

VGen is an open-source video synthesis codebase developed by the Tongyi Lab of Alibaba Group, featuring state-of-the-art video generative models. This repository includes implementations of the following methods:


- [I2VGen-xl: High-quality image-to-video synthesis via cascaded diffusion models](https://i2vgen-xl.github.io)
- [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://videocomposer.github.io)
- [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://higen-t2v.github.io)
- [A Recipe for Scaling up Text-to-Video Generation with Text-free Videos](https://tf-t2v.github.io)
- [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://instructvideo.github.io)
- [DreamVideo: Composing Your Dream Videos with Customized Subject and Motion](https://dreamvideo-t2v.github.io)
- [VideoLCM: Video Latent Consistency Model](https://arxiv.org/abs/2312.09109)
- [Modelscope text-to-video technical report](https://arxiv.org/abs/2308.06571)


VGen can produce high-quality videos from the input text, images, desired motion, desired subjects, and even the feedback signals provided.  It also offers a variety of commonly used video generation tools such as visualization, sampling, training, inference, join training using images and videos, acceleration, and more. 


<a href='https://i2vgen-xl.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2311.04145'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/damo-vilab/I2VGen-XL) [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2311.04145) 
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-a-discussion-sm-dark.svg)](https://huggingface.co/spaces/damo-vilab/I2VGen-XL/discussions) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/XUi0y7dxqEQ)  <a href='https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441039979087.mp4'><img src='source/logo.png'></a>
[![Replicate](https://replicate.com/cjwbw/i2vgen-xl/badge)](https://replicate.com/cjwbw/i2vgen-xl/)


## 🔥News!!!


- __[2024.06]__ We release the code and models of [InstructVideo](https://instructvideo.github.io/). InstructVideo enables the **LoRA** fine-tuning and inference in VGen. Feel free to use LoRA fine-tuning for other tasks.
- __[2024.04]__ We release the models of [DreamVideo](https://dreamvideo-t2v.github.io) and ModelScopeT2V V1.5!!! ModelScopeT2V V1.5 is further fine-tuned on ModelScopeT2V for 365k iterations with more data.
- __[2024.04]__ We release the code and models of [TF-T2V](https://tf-t2v.github.io)! 
- __[2024.04]__ We release the code and models of [VideoLCM](https://tf-t2v.github.io)! 
- __[2024.03]__ We release the training and inference code of [DreamVideo](https://dreamvideo-t2v.github.io)! 
- __[2024.03]__ We release the code and model of HiGen!!
- __[2024.01]__ The gradio demo of I2VGen-XL has been completed in [HuggingFace](https://huggingface.co/spaces/damo-vilab/I2VGen-XL), thanks to our colleague @[Wenmeng Zhou](https://github.com/wenmengzhou) and @[AK](https://twitter.com/_akhaliq) for the support, and welcome to try it out.
- __[2024.01]__ We support running the gradio app locally, thanks to our colleague @[Wenmeng Zhou](https://github.com/wenmengzhou) for the support and @[AK](https://twitter.com/_akhaliq) for the suggestion, and welcome to have a try.
- __[2024.01]__ Thanks @[Chenxi](https://chenxwh.github.io) for supporting the running of i2vgen-xl on [![Replicate](https://replicate.com/cjwbw/i2vgen-xl/badge)](https://replicate.com/cjwbw/i2vgen-xl/). Feel free to give it a try. 
- __[2024.01]__ The gradio demo of I2VGen-XL has been completed in [Modelscope](https://modelscope.cn/studios/damo/I2VGen-XL/summary), and welcome to try it out.
- __[2023.12]__ We have open-sourced the code and models for [DreamTalk](https://github.com/ali-vilab/dreamtalk), which can produce high-quality talking head videos across diverse speaking styles using diffusion models.
- __[2023.12]__ We release [TF-T2V](https://tf-t2v.github.io) that can scale up existing video generation techniques using text-free videos, significantly enhancing the performance of both [Modelscope-T2V](https://arxiv.org/abs/2308.06571) and [VideoComposer](https://videocomposer.github.io) at the same time.
- __[2023.12]__ We updated the codebase to support higher versions of xformer (0.0.22), torch2.0+, and removed the dependency on flash_attn.
- __[2023.12]__ We release [InstructVideo](https://instructvideo.github.io/) that can accept human feedback signals to improve VLDM
- __[2023.12]__ We release the diffusion based expressive talking head generation [DreamTalk](https://dreamtalk-project.github.io)
- __[2023.12]__ We release the high-efficiency video generation method [VideoLCM](https://arxiv.org/abs/2312.09109)
- __[2023.12]__ We release the code and model of [I2VGen-XL](https://i2vgen-xl.github.io) and the [ModelScope T2V](https://arxiv.org/abs/2308.06571)
- __[2023.12]__ We release the T2V method [HiGen](https://higen-t2v.github.io) and customizing T2V method [DreamVideo](https://dreamvideo-t2v.github.io).
- __[2023.12]__ We write an [introduction document](doc/introduction.pdf) for VGen and compare I2VGen-XL with SVD.
- __[2023.11]__ We release a high-quality I2VGen-XL model, please refer to the [Webpage](https://i2vgen-xl.github.io)


## TODO
- [x] Release the technical papers and webpage of [I2VGen-XL](doc/i2vgen-xl.md)
- [x] Release the code and pretrained models that can generate 1280x720 videos
- [x] Release the code and  models of [DreamTalk](https://github.com/ali-vilab/dreamtalk) that can generate expressive talking head
- [ ] Release the code and pretrained models of [HumanDiff]()
- [ ] Release models optimized specifically for the human body and faces
- [ ] Updated version can fully maintain the ID and capture large and accurate motions simultaneously
- [ ] Release other methods and the corresponding models


## Preparation

The main features of VGen are as follows:
- Expandability, allowing for easy management of your own experiments.
- Completeness, encompassing all common components for video generation.
- Excellent performance, featuring powerful pre-trained models in multiple tasks.


### Installation

```
conda create -n vgen python=3.8
conda activate vgen
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

You  also need to ensure that your system has installed the `ffmpeg` command. If it is not installed, you can install it using the following command:
```
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```


### Datasets

We have provided a **demo dataset** that includes images and videos, along with their lists in ``data``. 

*Please note that the demo images used here are for testing purposes and were not included in the training.*


### Clone the code

```
git clone https://github.com/ali-vilab/VGen.git
cd VGen
```


## Getting Started with VGen

### (1) Train your text-to-video model


Executing the following command to enable distributed training is as easy as that.
```
python train_net.py --cfg configs/t2v_train.yaml
```

In the `t2v_train.yaml` configuration file, you can specify the data, adjust the video-to-image ratio using `frame_lens`, and validate your ideas with different Diffusion settings, and so on.

- Before the training, you can download any of our open-source models for initialization. Our codebase supports custom initialization and `grad_scale` settings, all of which are included in the `Pretrain` item in yaml file.
- During the training, you can view the saved models and intermediate inference results in the `workspace/experiments/t2v_train`directory.

After the training is completed, you can perform inference on the model using the following command.
```
python inference.py --cfg configs/t2v_infer.yaml
```
Then you can find the videos you generated in the `workspace/experiments/test_img_01` directory. For specific configurations such as data, models, seed, etc., please refer to the `t2v_infer.yaml` file.


*If you want to directly load our previously open-sourced [Modelscope T2V model](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis/tree/main), please refer to [this link](https://github.com/damo-vilab/i2vgen-xl/issues/31).*


<!-- <table>
<center>
  <tr>
    <td ><center>
      <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441754174077.mp4"></video>	
    </center></td>
    <td ><center>
      <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441138824052.mp4"></video>	
    </center></td>
  </tr>
</center>
</table>
</center> -->




### (2) Run the I2VGen-XL model

(i) Download model and test data:
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('damo/I2VGen-XL', cache_dir='models/', revision='v1.0.0')
```

or you can also download it through HuggingFace (https://huggingface.co/damo-vilab/i2vgen-xl):
```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/damo-vilab/i2vgen-xl
```


(ii) Run the following command:
```
python inference.py --cfg configs/i2vgen_xl_infer.yaml
```
or you can run:
```
python inference.py --cfg configs/i2vgen_xl_infer.yaml  test_list_path data/test_list_for_i2vgen.txt test_model models/i2vgen_xl_00854500.pth
```
The `test_list_path` represents the input image path and its corresponding caption. Please refer to the specific format and suggestions within demo file `data/test_list_for_i2vgen.txt`. `test_model` is the path for loading the model. In a few minutes, you can retrieve the high-definition video you wish to create from the `workspace/experiments/test_list_for_i2vgen` directory. At present, we find that the current model performs inadequately on **anime images** and **images with a black background** due to the lack of relevant training data. We are consistently working to optimize it.


(iii) Run the gradio app locally:
```
python gradio_app.py
```

(iv) Run the model on ModelScope and HuggingFace:
- [Modelscope](https://modelscope.cn/studios/damo/I2VGen-XL/summary)
- [HuggingFace](https://huggingface.co/spaces/damo-vilab/I2VGen-XL)


<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HRER' below to view the original video.</span>

<center>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01CCEq7K1ZeLpNQqrWu_!!6000000003219-0-tps-1280-720.jpg"></image>	
    </center></td>
    <td ><center>
      <!-- <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442125067544.mp4"></video>	 -->
      <image  height="260" src="https://img.alicdn.com/imgextra/i4/O1CN01hIQcvG1spmQMLqBo0_!!6000000005816-1-tps-1280-704.gif"></image>	
    </center></td>
  </tr> 
  <tr>
    <td ><center>
      <p>Input Image</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442125067544.mp4">HERE</a> to view the generated video.</p>
    </center></td>
  </tr> 
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i4/O1CN01ZXY7UN23K8q4oQ3uG_!!6000000007236-2-tps-1280-720.png"></image>	
    </center></td>
    <td ><center>
      <!-- <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441385957074.mp4"></video>	 -->
      <image height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01iaSiiv1aJZURUEY53_!!6000000003309-1-tps-1280-704.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Input Image</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441385957074.mp4">HERE</a> to view the generated video.</p>
    </center></td>
  </tr> 
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i3/O1CN01NHpVGl1oat4H54Hjf_!!6000000005242-2-tps-1280-720.png"></image>	
    </center></td>
    <td ><center>
      <!-- <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442102706767.mp4"></video>	 -->
      <!-- <image muted="true" height="260" src="https://img.alicdn.com/imgextra/i4/O1CN01DgLj1T240jfpzKoaQ_!!6000000007329-1-tps-1280-704.gif"></image>	
       -->
      <image  height="260" src="https://img.alicdn.com/imgextra/i4/O1CN01DgLj1T240jfpzKoaQ_!!6000000007329-1-tps-1280-704.gif"></image>
    </center></td>
  </tr> 
  <tr>
    <td ><center>
      <p>Input Image</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442102706767.mp4">HERE</a> to view the generated video.</p>
    </center></td>
  </tr> 
  <tr>
    <td ><center>
      <image height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01odS61s1WW9tXen21S_!!6000000002795-0-tps-1280-720.jpg"></image>	
    </center></td>
    <td ><center>
      <!-- <video muted="true" autoplay="true" loop="true" height="260" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442163934688.mp4"></video>	 -->
      <image height="260" src="https://img.alicdn.com/imgextra/i3/O1CN01Jyk1HT28JkZtpAtY6_!!6000000007912-1-tps-1280-704.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Input Image</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/442163934688.mp4">HERE</a> to view the generated video.</p>
    </center></td>
  </tr> 
</center>
</table>
</center>

(ii) Run the following command:
```
python inference.py --cfg configs/i2vgen_xl_train.yaml
```
In a few minutes, you can retrieve the high-definition video you wish to create from the `workspace/experiments/test_img_01` directory. At present, we find that the current model performs inadequately on **anime images** and **images with a black background** due to the lack of relevant training data. We are consistently working to optimize it.


### (3) Run the HiGen model

(i) Download model:
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/HiGen', cache_dir='models/')
```
Then you might need the following command to move the checkpoints to the "models/" directory:
```
mv ./models/iic/HiGen/* ./models/
```

(ii) Run the following command for text-to-video generation:
```
python inference.py --cfg configs/higen_infer.yaml
```
In a few minutes, you can retrieve the videos you wish to create from the `workspace/experiments/text_list_for_t2v_share` directory.
Then you can execute the following command to perform super-resolution on the generated videos:
```
python inference.py --cfg configs/sr600_infer.yaml
```
Finally, you can retrieve the high-definition video from the `workspace/experiments/text_list_for_t2v_share` directory.

<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HERE' below to view the original video.</span>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="source/duck.png"></image>	
    </center></td>
    <td ><center>
      <image height="260" src="source/bat_man.png"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/452227605224.mp4">HERE</a> to view the generated video.</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/452015792863.mp4">HERE</a> to view the generated video.</p>
    </center></td>
  </tr>
</center>
</table>
</center>


### (4) DreamVideo
Our DreamVideo uses `ModelScopeT2V V1.5` as the base video diffusion model. `ModelScopeT2V V1.5 is further fine-tuned on ModelScopeT2V for 365k iterations with more data.`

#### Download ModelScopeT2V V1.5 and adapter weights of DreamVideo
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/dreamvideo-t2v', cache_dir='models/')
```
Then you might need the following command to move the checkpoints to the "models/" directory:
```
mv ./models/iic/dreamvideo-t2v/* ./models/
```
Or you can download the checkpoint of ModelScopeT2V V1.5 and adapter weights of DreamVideo from this [link](https://modelscope.cn/models/iic/dreamvideo-t2v/files). 

#### Training
(i) Subject Learning

Step 1: learn a textual identity using Textual Inversion.
```
python train_net.py --cfg configs/dreamvideo/subjectLearning/dog2_subjectLearning_step1.yaml
```

Step 2: train an identity adapter by incorporating the learned textual identity.
```
python train_net.py --cfg configs/dreamvideo/subjectLearning/dog2_subjectLearning_step2.yaml
```

Tips:
- Generally, step 1 takes `1500 to 3000` training steps, and step 2 takes `500 to 1000` training steps. For certain subjects (like cats, etc.), excessive training may generate unnatural videos, and using text embedding with fewer training steps or reducing the training steps of step 2 may help.
- For some subjects (like dogs, etc.), setting `use_mask_diffusion` to `True` may achieve better results. Make sure to put the binary masks of the subject into the folder `data/images/custom/YOUR_SUBJECT/masks`, and you can use [SAM](https://github.com/facebookresearch/segment-anything) to obtain these masks.

(ii) Motion Learning

Train a motion adapter on the given videos.
```
python train_net.py --cfg configs/dreamvideo/motionLearning/carTurn_motionLearning.yaml
```

You can customize your own configuration files for subject/motion learning.

Tips:
- Generally, motion learning takes `500 to 2000` training steps.
- Try setting `p_image_zero` from `0 to 0.5` to adjust the effect of appearance guidance during training.
- Try increasing training steps or increasing the learning rate for single video motion customization to better align the motion pattern.


#### Inference
(i) Subject Customization
```
python inference.py --cfg configs/dreamvideo/infer/subject_dog2.yaml
```

(ii) Motion Customization
```
python inference.py --cfg configs/dreamvideo/infer/motion_carTurn.yaml
```
For inference with appearance guidance, make sure to add images of foreground objects (e.g., any image of a bear) to the folder `data/images/motionReferenceImgs` and modify your test file.

Tips:
- Try setting `appearance_guide_strength_cond` and `appearance_guide_strength_uncond` from `0 to 1` to adjust the effect of appearance guidance during inference.
- We do not use DDIM Inversion by default. However, for single video motion customization, you can try setting `inverse_noise_strength` to `0~0.5` to better align the training video. For multi-video motion customization, we recommend setting `inverse_noise_strength` to `0`.


(iii) Joint Customization
```
python inference.py --cfg configs/dreamvideo/infer/joint_dog2_carTurn.yaml
```

Tips:
- Try changing `identity_adapter_index` and `motion_adapter_index` for better results. Typically, increasing identity_adapter_index improves identity preservation, while increasing motion_adapter_index enhances motion alignment. Balance the two for optimal results.


#### Examples
We provide some examples for inference. Before you start, make sure you download the models.

(i) Subject Customization
```
python inference.py --cfg configs/dreamvideo/infer/examples/subject_dog2.yaml

python inference.py --cfg configs/dreamvideo/infer/examples/subject_wolf_plushie.yaml
```

<center>
<table>
<center>
<tr>
  <td style="text-align:center;"><center><b>Subject</b></center></td>
  <td style="text-align:center;"><center><b>Generated Video</b></center></td>
  <td style="text-align:center;"><center><b>Subject</b></center></td>
  <td style="text-align:center;"><center><b>Generated Video</b></center></td>
</tr>
  <tr>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01wjCCkF1f7jMzZh6pn_!!6000000003960-2-tps-256-256.png" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i2/O1CN016RJQ1u1zEPSfSzfjh_!!6000000006682-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN013ekGAg1vwbDDt8N9H_!!6000000006237-2-tps-256-256.png" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i2/O1CN011HQwSb1REglZ81Ki8_!!6000000002080-1-tps-256-256.gif" />
    </center></td>
  </tr>
  <tr>
  <td style="text-align:center;"><center>dog</center></td>
  <td style="text-align:center;"><center>"a * eating pizza"</br> seed: 2767</center></td>
  <td style="text-align:center;"><center>wolf plushie</center></td>
  <td style="text-align:center;"><center>"a * running in the forest" </br> seed: 2339</center></td>
</tr>
</center>
</table>
</center>

(ii) Motion Customization
```
python inference.py --cfg configs/dreamvideo/infer/examples/motion_carTurn.yaml

python inference.py --cfg configs/dreamvideo/infer/examples/motion_playingGuitar.yaml
```

<center>
<table>
<center>
<tr>
  <td style="text-align:center;"><center><b>Motion</b></center></td>
  <td style="text-align:center;"><center><b>Generated Video</b></center></td>
  <td style="text-align:center;"><center><b>Motion</b></center></td>
  <td style="text-align:center;"><center><b>Generated Video</b></center></td>
</tr>
  <tr>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i1/O1CN01r9yeiR1irw83xSXZz_!!6000000004467-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01OKbdZj1sy1b2Nhq5M_!!6000000005834-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01GkeSm31p1S679Hf3J_!!6000000005300-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01okWw0V1s544J9v6BU_!!6000000005714-1-tps-256-256.gif" />
    </center></td>
  </tr>
  <tr>
  <td style="text-align:center;"><center>"a car running on the road"</center></td>
  <td style="text-align:center;"><center>"a lion running on the road"</br> seed: 8888</center></td>
  <td style="text-align:center;"><center>"a person is playing guitar"</center></td>
  <td style="text-align:center;"><center>"a monkey is playing guitar on Mars" </br> seed: 8888</center></td>
</tr>
</center>
</table>
</center>

(iii) Joint Customization
```
python inference.py --cfg configs/dreamvideo/infer/examples/joint_dog2_carTurn.yaml

python inference.py --cfg configs/dreamvideo/infer/examples/joint_dog2_playingGuitar.yaml

python inference.py --cfg configs/dreamvideo/infer/examples/joint_wolf_plushie_carTurn.yaml

python inference.py --cfg configs/dreamvideo/infer/examples/joint_wolf_plushie_playingGuitar.yaml
```

<center>
<table>
<center>
  <tr>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01uT4pI71EAwWGLx9hP_!!6000000000312-2-tps-256-256.png" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01wjCCkF1f7jMzZh6pn_!!6000000003960-2-tps-256-256.png" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN013ekGAg1vwbDDt8N9H_!!6000000006237-2-tps-256-256.png" />
    </center></td>
  </tr>
  <tr >
    <td style="text-align:center;"></td>
    <td style="text-align:center;"><center>dog</center></td>
    <td style="text-align:center;"><center>wolf plushie</center></td>
  </tr>
  <tr>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i1/O1CN01r9yeiR1irw83xSXZz_!!6000000004467-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01GOVb6p244rHKdTzPJ_!!6000000007338-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i3/O1CN01uwNzW21kLXO82PKjp_!!6000000004667-1-tps-256-256.gif" />
    </center></td>
  </tr>
  <tr>
    <td style="text-align:center;"><center>"a car running on the road"</center></td>
    <td style="text-align:center;"><center>"a * running on the beach"</br> seed: 8888</center></td>
    <td style="text-align:center;"><center>"a * running on the road"</br> seed: 3677</center></td>
  </tr>
  <tr>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01GkeSm31p1S679Hf3J_!!6000000005300-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i1/O1CN01ouNpOm1Ptn4sG4uFV_!!6000000001899-1-tps-256-256.gif" />
    </center></td>
    <td><center>
      <img src="https://img.alicdn.com/imgextra/i4/O1CN01q7kZra1SpcJezWkrk_!!6000000002296-1-tps-256-256.gif" />
    </center></td>
  </tr>
  <tr>
    <td style="text-align:center;"><center>"a person is playing guitar"</center></td>
    <td style="text-align:center;"><center>"a * is playing guitar on the moon"</br> seed: 8888</center></td>
    <td style="text-align:center;"><center>"a * is playing guitar"</br> seed: 6071</center></td>
  </tr>
</center>
</table>
</center>


### (5) Run the TF-T2V (CVPR-2024) model

(i) Download model:
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/tf-t2v', cache_dir='models/')
```
Then you might need the following command to move the checkpoints to the "models/" directory:
```
mv ./models/iic/tf-t2v/* ./models/
```

(ii) We provide a config file for generating 16-frame video with 448x256 resolution. The command is as follows:
```
python inference.py --cfg configs/tft2v_t2v_infer.yaml
```
(If there are environmental problems during operation, we also provide the environment configuration "tft2v_environment.yaml" of TF-T2V for your reference.)


In a few minutes, you can retrieve the videos you wish to create from the `workspace/experiments/text_list_for_tft2v` directory.
Then you can execute the following command to perform super-resolution on the generated videos:
```
python inference.py --cfg configs/tft2v_16frames_sr600_infer.yaml
```
Finally, you can retrieve the high-definition video from the `workspace/experiments/text_list_for_tft2v` directory.
(It should be noted that the super-resolution model only supports 32-frame input, and 16-frame video cannot be used, thus we construct a pseudo 32-frame video by copying frames.)

<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HERE' below to view the original video.</span>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i1/O1CN014OJopR1MFXQ0y9jN1_!!6000000001405-1-tps-320-180.gif"></image>	
    </center></td>
    <td ><center>
      <image height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01gzSvmL1s7oRR4sBJf_!!6000000005720-1-tps-320-180.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/d0pnU3FxV01sT3ljVGowRFVmeVJOR2R0U3Rya05udGJoZ29BR203QzlURzZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/d0pnU3FxV01sT3ljVGowRFVmeVJOR2FXNTVHdDdoTndIOHh1R09zejZsQzZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
  </tr>
</center>
</table>
</center>

(iii) Additionally, you can run the following command for text-to-video generation (32 frames):
```
python inference.py --cfg configs/tft2v_t2v_32frames_infer.yaml
```

In a few minutes, you can retrieve the videos you wish to create from the `workspace/experiments/text_list_for_tft2v_32frame` directory.
Then you can execute the following command to perform super-resolution on the generated videos:
```
python inference.py --cfg configs/tft2v_32frames_sr600_infer.yaml
```
Finally, you can retrieve the high-definition video from the `workspace/experiments/text_list_for_tft2v_32frame` directory.
(It should be noted that the super-resolution model only supports 32-frame input, and 16-frame video cannot be used.)

<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HERE' below to view the original video.</span>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i3/O1CN01PN4Gv31ZfGfrf4bI3_!!6000000003221-1-tps-320-180.gif"></image>	
    </center></td>
    <td ><center>
      <image height="260" src="https://img.alicdn.com/imgextra/i3/O1CN01qMPlvb26JAtAqovud_!!6000000007640-1-tps-320-180.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/eGhWQVU2UHBJcWRTMXRISFNKcFhnY0Z1dTlkM013M3ZsNDZCUHlad2lrRzZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/eGhWQVU2UHBJcWRTMXRISFNKcFhnWDBaTmQxYTc1TzRYS3BFTG1TMXhORzZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
  </tr>
</center>
</table>
</center>


(iv) Run the following command for compositional video generation like videocomposer (32 frames):

```
python inference.py --cfg configs/tft2v_vcomposer_32frames_infer.yaml
```
In a few minutes, you can retrieve the videos you wish to create from the `workspace/experiments/vid_list_vcomposer_32frame` directory.
Then you can execute the following command to perform super-resolution on the generated videos:
```
python inference.py --cfg configs/tft2v_vcomposer_32frames_sr600_infer.yaml
```
Finally, you can retrieve the high-definition video from the `workspace/experiments/vid_list_vcomposer_32frame` directory.


<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HERE' below to view the original video.</span>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i2/O1CN01I4cVz01eHWDQKnC0b_!!6000000003846-1-tps-320-180.gif"></image>	
    </center></td>
    <td ><center>
      <image height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01VWLCS01TEo65Rkv8F_!!6000000002351-1-tps-320-180.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/dkwzYUtUNExsSzQ2cGZRM0N3Z0VYWlczOW5YanFzdkkzYW1hRDZRRGlHNjZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/dkwzYUtUNExsSzQ2cGZRM0N3Z0VYUlNNRHVwN0ZmbndFMkpZeFlEczdsZTZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
  </tr>
</center>
</table>
</center>


(v) We also provide a config file for generating 16-frame video with 448x256 resolution under the compositional video synthesis setting. The command is as follows:
```
python inference.py --cfg configs/tft2v_vcomposer_infer.yaml
```
You can also generate a 16-frame video with 896x512 resolution within one model by running:
```
python inference.py --cfg configs/tft2v_vcomposer_896x512_infer.yaml
```

It should be noted that the super-resolution model only supports 32-frame input, and 16-frame video cannot be used.



### (6) Run the VideoLCM model

(i) Download models as in TF-T2V (if you have already downloaded them in TF-T2V, skip this step):
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/tf-t2v', cache_dir='models/')
```
Then you might need the following command to move the checkpoints to the "models/" directory:
```
mv ./models/iic/tf-t2v/* ./models/
```

(ii) Run the following command for text-to-video generation (16 frames with 448x256 resolution):
```
python inference.py --cfg configs/videolcm_t2v_infer.yaml
```

To generate high-resolution videos (1280x720 resolution), you can run the following command:
```
python inference.py --cfg configs/videolcm_t2v_16frames_sr600_infer.yaml
```


<span style="color:red">Due to the compression of our video quality in GIF format, please click 'HERE' below to view the original video.</span>
<table>
<center>
  <tr>
    <td ><center>
      <image  height="260" src="https://img.alicdn.com/imgextra/i2/O1CN01AqHVYW1T2tnVCfd1u_!!6000000002325-1-tps-320-180.gif"></image>	
    </center></td>
    <td ><center>
      <image height="260" src="https://img.alicdn.com/imgextra/i1/O1CN01a7YqU51GxbpoqiKsv_!!6000000000689-1-tps-320-180.gif"></image>	
    </center></td>
  </tr>
  <tr>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/TGdxZ2dwem8xVXFtcm1TYjlCMXd6UXFzMVM/dDV1MTltVldlWlVUajlkcTZQZWw1SnpKVVVCTlh4OVFON0V5UUVMUDduY1RJak82VE1sdXdHTjNOaHc9PQ">HERE</a> to view the generated video.</p>
    </center></td>
    <td ><center>
      <p>Click <a href="https://cloud.video.taobao.com/vod/play/OVZ5THVhZW1hbkZxSm4wRzVNNm0xMDFxbjdoVzR6VjhXWlBoSkNjKw/VjI2UGVsNUp6SlVVQk5YeDlRTjdFeVFFTFA3bmNUSWpPNlRNbHV3R04zTmh3PT0">HERE</a> to view the generated video.</p>
    </center></td>
  </tr>
</center>
</table>
</center>


(iii) Run the following command for compositional video generation (16 frames with 448x256 resolution):
```
python inference.py --cfg configs/videolcm_vcomposer_infer.yaml
```

### (7) InstructVideo (CVPR 2024)

Feel free to reach out (hj.yuan@zju.edu.cn) if have questions.

#### Dataset preparation and environment configuration
The training of InstructVideo requires video-text pairs to save computational cost during reward fine-tuning.
In the paper, we utilize a small set of videos in WebVid to fine-tune our base model.
The file list is shown under the folder:
```
data/instructvideo/webvid_simple_animals_2_selected_20_train_file_list/00000.txt
```
You should try filtering the videos from your webvid dataset to compose the training data. Another alternative is to use your own video-text pairs.
(I tested InstructVideo on WebVid data and some proprietary data. Both worked.)

Concerning the environment configuration, you should follow the instructions for [VGen installation](https://github.com/ali-vilab/VGen?tab=readme-ov-file#installation).


#### Pre-trained weights preparation
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/InstructVideo', cache_dir='models/')
```
You need to move the checkpoints to the "models/" directory:
```
mv ./models/iic/InstructVideo/* ./models/
```
Note that `models/model_scope_v1-4_0600000.pth` is the pre-trained base model used in the paper.
The fine-tuned model is placed under the folder `models/instructvideo-finetuned`.

You can get access to the provided files on [Instructvideo ModelScope Page](https://modelscope.cn/models/iic/InstructVideo/files).


#### The inference of InstructVideo
You can leverage the provided fine-tuned checkpoints to generate videos by running the command:
```
bash configs/instructvideo/eval_generate_videos.sh
```
This command uses yaml files under `configs/instructvideo/eval`, containing caption file paths for generating videos of in-domain animals, new animals and non-animals.
Feel free to switch among them or replace them with your own captions.
Although we fine-tuned using 20-step DDIM, you can still use 50-step DDIM generation.

#### The reward fine-tuning of InstructVideo
You can perform InstrcutVideo reward fine-tuning by running the command:
```
bash configs/instructvideo/train.sh
```
Since performing reward fine-tuning can lead to over-optimization, I strongly recommend checking the generation performance on some evaluation captions regularly (like the captions indicated in `configs/instructvideo/eval`).


### (8) Other methods

In preparation!!


## Customize your own approach

Our codebase essentially supports all the commonly used components in video generation. You can manage your experiments flexibly by adding corresponding registration classes, including `ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, VISUAL, DIFFUSION, PRETRAIN`, and can be compatible with all our open-source algorithms according to your own needs. If you have any questions, feel free to give us your feedback at any time.



## BibTeX

If this repo is useful to you, please cite our corresponding technical paper.


```bibtex
@article{wang2023videocomposer,
  title={Videocomposer: Compositional Video Synthesis with Motion Controllability},
  author={Wang, Xiang and Yuan, Hangjie and Zhang, Shiwei and Chen, Dayou and Wang, Jiuniu and Zhang, Yingya and Shen, Yujun and Zhao, Deli and Zhou, Jingren},
  journal={NeurIPS},
  volume={36},
  year={2023}
}
@article{2023i2vgenxl,
  title={I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models},
  author={Zhang, Shiwei and Wang, Jiayu and Zhang, Yingya and Zhao, Kang and Yuan, Hangjie and Qing, Zhiwu and Wang, Xiang  and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2311.04145},
  year={2023}
}
@article{wang2023modelscope,
  title={Modelscope text-to-video technical report},
  author={Wang, Jiuniu and Yuan, Hangjie and Chen, Dayou and Zhang, Yingya and Wang, Xiang and Zhang, Shiwei},
  journal={arXiv preprint arXiv:2308.06571},
  year={2023}
}
@inproceedings{dreamvideo,
  title={DreamVideo: Composing Your Dream Videos with Customized Subject and Motion},
  author={Wei, Yujie and Zhang, Shiwei and Qing, Zhiwu and Yuan, Hangjie and Liu, Zhiheng and Liu, Yu and Zhang, Yingya and Zhou, Jingren and Shan, Hongming},
  booktitle={CVPR},
  year={2024}
}
@inproceedings{higen,
  title={Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation},
  author={Qing, Zhiwu and Zhang, Shiwei and Wang, Jiayu and Wang, Xiang and Wei, Yujie and Zhang, Yingya and Gao, Changxin and Sang, Nong },
  booktitle={CVPR},
  year={2024}
}
@article{wang2023videolcm,
  title={VideoLCM: Video Latent Consistency Model},
  author={Wang, Xiang and Zhang, Shiwei and Zhang, Han and Liu, Yu and Zhang, Yingya and Gao, Changxin and Sang, Nong },
  journal={arXiv preprint arXiv:2312.09109},
  year={2023}
}
@article{ma2023dreamtalk,
  title={DreamTalk: When Expressive Talking Head Generation Meets Diffusion Probabilistic Models},
  author={Ma, Yifeng and Zhang, Shiwei and Wang, Jiayu and Wang, Xiang and Zhang, Yingya and Deng Zhidong},
  journal={arXiv preprint arXiv:2312.09767},
  year={2023}
}
@inproceedings{InstructVideo,
  title={InstructVideo: Instructing Video Diffusion Models with Human Feedback},
  author={Yuan, Hangjie and Zhang, Shiwei and Wang, Xiang and Wei, Yujie and Feng, Tao and Pan, Yining and Zhang, Yingya and Liu, Ziwei and Albanie, Samuel and Ni, Dong},
  booktitle={CVPR},
  year={2024}
}
@inproceedings{TFT2V,
  title={A Recipe for Scaling up Text-to-Video Generation with Text-free Videos},
  author={Wang, Xiang and Zhang, Shiwei and Yuan, Hangjie and Qing, Zhiwu and Gong, Biao and Zhang, Yingya and Shen, Yujun and Gao, Changxin and Sang, Nong},
  booktitle={CVPR},
  year={2024}
}
```


## Acknowledgement

We would like to express our gratitude for the contributions of several previous works to the development of VGen. This includes, but is not limited to [Composer](https://arxiv.org/abs/2302.09778), [ModelScopeT2V](https://modelscope.cn/models/damo/text-to-video-synthesis/summary), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [OpenCLIP](https://github.com/mlfoundations/open_clip), [WebVid-10M](https://m-bain.github.io/webvid-dataset/), [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), [Pidinet](https://github.com/zhuoinoulu/pidinet) and [MiDaS](https://github.com/isl-org/MiDaS). We are committed to building upon these foundations in a way that respects their original contributions.



## Disclaimer

This open-source model is trained with using [WebVid-10M](https://m-bain.github.io/webvid-dataset/) and [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) datasets and is intended for <strong>RESEARCH/NON-COMMERCIAL USE ONLY</strong>. 
