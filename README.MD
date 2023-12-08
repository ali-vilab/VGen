# VGen


![figure1](source/VGen.jpg "figure1")

VGen is an open-source video synthesis codebase developed by the Tongyi Lab of Alibaba Group, featuring state-of-the-art video generative models. This repository includes implementations of the following methods:

- [I2VGen-xl: High-quality image-to-video synthesis via cascaded diffusion models](https://i2vgen-xl.github.io/)
- [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://videocomposer.github.io/)
- [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://higen-t2v.github.io/)
- [A Recipe for Scaling up Text-to-Video Generation with Text-free Videos]()
- [InstructVideo: Instructing Video Diffusion Models with Human Feedback]()
- [DreamVideo: Composing Your Dream Videos with Customized Subject and Motion](https://dreamvideo-t2v.github.io/)
- [VideoLCM: Video Latent Consistency Model]()
- [Modelscope text-to-video technical report](https://arxiv.org/abs/2308.06571)


VGen can produce high-quality videos from the input text, images, desired motion, desired subjects, and even the feedback signals provided.  It also offers a variety of commonly used video generation tools such as visualization, sampling, training, inference, join training using images and videos, acceleration, and more. 

<a href='https://i2vgen-xl.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2311.04145'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/XUi0y7dxqEQ) <a href='https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/441039979087.mp4'><img src='source/logo.png'></a>


## 🔥News!!!
- __[2023.12]__ We release the code and model of I2VGen-XL (Soon)
- __[2023.12]__ We write an [introduction docment](doc/introuctions.pdf) for VGen and compare I2VGen-XL with SVD.
- __[2023.11]__ We release a high-quality I2VGen-XL model, please refer to the [Webpage](https://i2vgen-xl.github.io)


## TODO
- [x] Release the technical papers and webpage of [I2VGen-XL](doc/i2vgen-xl.md)
- [ ] Release the code and pretrained models that can generate 1280x720 videos
- [ ] Release models optimized specifically for the human body and faces
- [ ] Updated version can fully maintain the ID and capture large and accurate motions simultaneously
- [ ] Release other methods and the corresponding models


## Installation:

Requirements:
- Python==3.8
- ffmpeg (for motion vector extraction)
- torch==1.12.0+cu113
- torchvision==0.13.0+cu113
- open-clip-torch==2.0.2
- transformers==4.18.0
- flash-attn==0.2 
- xformers==0.0.13
- motion-vector-extractor==1.0.6 (for motion vector extraction)

You also can create the same environment as ours with the following command:
```
conda env create -f environment.yaml
```

## Getting Started with VGen

Come soon.


## BibTeX

If this repo is useful to you, please cite our technical paper.


```bibtex
@article{2023i2vgenxl,
  title={I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models},
  author={Zhang, Shiwei and Wang, Jiayu and Zhang, Yingya and Zhao, Kang and Yuan, Hangjie and Qing, Zhiwu and Wang, Xiang  and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2311.04145},
  year={2023}
}
@article{2023videocomposer,
  title={VideoComposer: Compositional Video Synthesis with Motion Controllability},
  author={Wang, Xiang and Yuan, Hangjie and Zhang, Shiwei and Chen, Dayou and Wang, Jiuniu, and Zhang, Yingya, and Shen, Yujun, and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2306.02018},
  year={2023}
}
@article{wang2023modelscope,
  title={Modelscope text-to-video technical report},
  author={Wang, Jiuniu and Yuan, Hangjie and Chen, Dayou and Zhang, Yingya and Wang, Xiang and Zhang, Shiwei},
  journal={arXiv preprint arXiv:2308.06571},
  year={2023}
}
@article{dreamvideo,
	title={DreamVideo: Composing Your Dream Videos with Customized Subject and Motion},
	author={Wei, Yujie and Zhang, Shiwei and Qing, Zhiwu and Yuan, Hangjie and Liu, Zhiheng and Liu, Yu and Zhang, Yingya and Zhou, Jingren and Shan, Hongming},
	journal={arXiv preprint arXiv:2312.04433},
	year={2023}
}
@article{qing2023higen,
  title={Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation},
  author={Qing, Zhiwu and Zhang, Shiwei and Wang, Jiayu and Wang, Xiang and Wei, Yujie and Zhang, Yingya and Gao, Changxin and Sang, Nong },
  journal={arXiv preprint arXiv:2312.04483},
  year={2023}
}
```
