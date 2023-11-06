# I2VGen-XL

Official repo for [I2VGen-XL: High-Quality Image-to-Video Synthesis Via Cascaded Diffusion Models]().


Please visit the [Project Page](https://i2vgen-xl.github.io/index.html) for more examples.



![figure1](source/fig_01.jpg "figure1")



I2VGen-XL is capable of generating high-quality, realistically animated, and temporally coherent high-definition videos from a single static image based on user input.


*The initial version of our model has been open-sourced at [ModelScope](https://modelscope.cn/models/damo/Image-to-Video/summary). We are now enhancing this version, with a particular focus on improving capabilities for modeling motion and semantic.*




## TODO
- [x] Release the technical paper and the webpage.
- [ ] Release the code and pre-trained models that can generate 1280x720 videos.
- [ ] Release models optimized specifically for human bodies and faces.
- [ ] Release a version that can fully maintain the ID and capture large and accurate motions simultaneously.


*We are committed to continuously enhancing the model's performance and will keep open-sourcing our updates. Your support and interest are greatly appreciated.*


## Method

![method](source/fig_02.jpg "method")


## Examples

![figure2](source/fig_04.png "figure2")


## Running the Model



## BibTeX

If this repo is useful to you, please cite our technical paper.


```bibtex
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
