# I2VGen-XL

Official repo for [I2VGen-XL: High-Quality Image-to-Video Synthesis Via Cascaded Diffusion Models]()


Please see [Project Page](https://i2vgen-xl.github.io/index.html) for more examples.


![figure1](source/fig_01.jpg "figure1")



I2VGen-XL is capable of generating high-quality, realistically animated, and temporally coherent high-definition videos from a single input static image, based on user input.



*Our initial version has already been open-sourced on [Modelscope](https://modelscope.cn/models/damo/Image-to-Video/summary). This project focuses on improving the version, especially in terms of motions and semantics.*



## TODO
- [x] Release the technical papers and webpage
- [ ] Release the code and pretrained models that can generate 1280x720 videos
- [ ] Release models optimized specifically for human body and faces
- [ ] Updated version can fully maintain the ID and capture large and accurate motions simultaneously



*In the future, we will continue to enhance the model's performance and open-source it here. Your support and attention are welcome.*

## Method

![method](source/fig_02.jpg "method")


## Examples

![figure2](source/fig_04.png "figure2")


## Running by Yourself



## BibTeX

If this repo is useful to you, please cite our technical paper.


```bibtex
@article{2023i2vgenxl,
  title={I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models},
  author={Zhang, Shiwei*  and Wang, Jiayu* and Zhang, Yingya* and Zhao, Kang and Yuan, Hangjie and Qing, Zhiwu and Wang, Xiang  and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2311.04145},
  year={2023}
}
@article{2023videocomposer,
  title={VideoComposer: Compositional Video Synthesis with Motion Controllability},
  author={Wang, Xiang* and Yuan, Hangjie* and Zhang, Shiwei* and Chen, Dayou* and Wang, Jiuniu, and Zhang, Yingya, and Shen, Yujun, and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2306.02018},
  year={2023}
}
@article{wang2023modelscope,
  title={Modelscope text-to-video technical report},
  author={Wang, Jiuniu and Yuan, Hangjie and Chen, Dayou and Zhang, Yingya and Wang, Xiang and Zhang, Shiwei},
  journal={arXiv preprint arXiv:2308.06571},
  year={2023}
}
```
