<div align="center">
<h1> InstructVideo: Instructing Video Diffusion Models<br>with Human Feedback
</h1>

<div>
    <a href='https://jacobyuan7.github.io/' target='_blank'>Hangjie Yuan</a>&emsp;
    <a href='https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ&hl=en&oi=ao' target='_blank'>Shiwei Zhang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=cQbXvkcAAAAJ&hl=en' target='_blank'>Xiang Wang</a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=grn93WcAAAAJ' target='_blank'>Yujie Wei</a>&emsp;
    <a href='https://scholar.google.com/citations?user=JT8hRbgAAAAJ&hl=en' target='_blank'>Tao Feng</a>&emsp;<br>
<!--     Yining Pan&emsp;<br> -->
    <a href='https://pynsigrid.github.io/' target='_blank'>Yining Pan</a>&emsp;
    <a href='https://scholar.google.com/citations?user=16RDSEUAAAAJ&hl=en' target='_blank'>Yingya Zhang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>&emsp;
    <a href='https://samuelalbanie.com/' target='_blank'>Samuel Albanie</a>&emsp;
    <a href='https://scholar.google.com/citations?user=boUZ-jwAAAAJ&hl=en' target='_blank'>Dong Ni</a>&emsp;
</div>
<br>


[![arXiv](https://img.shields.io/badge/arXiv-InstructVideo-<COLOR>.svg)](xxxxx)
[![Project Page](https://img.shields.io/badge/Project_Page-InstructVideo-<COLOR>.svg)](https://instructvideo.github.io/)
[![GitHub Stars](https://img.shields.io/github/stars/damo-vilab/i2vgen-xl?style=social)](https://github.com/damo-vilab/i2vgen-xl)
[![GitHub Forks](https://img.shields.io/github/forks/damo-vilab/i2vgen-xl)](https://github.com/damo-vilab/i2vgen-xl)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdamo-vilab%2Fi2vgen-xl&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</div>


> Abstract:
> Diffusion models have emerged as the de facto paradigm for video generation. 
> However, their reliance on web-scale data of varied quality often yields results that are visually unappealing and misaligned with the textual prompts.
> To tackle this problem, we propose InstructVideo to instruct text-to-video diffusion models with human feedback by reward fine-tuning.
> InstructVideo has two key ingredients: 1) To ameliorate the cost of reward fine-tuning induced by generating through the full DDIM sampling chain, we recast reward fine-tuning as editing. By leveraging the diffusion process to corrupt a sampled video, InstructVideo requires only partial inference of the DDIM sampling chain, reducing fine-tuning cost while improving fine-tuning efficiency. 2) To mitigate the ab sence of a dedicated video reward model for human preferences, we repurpose established image reward models, e.g., HPSv2. 
> To this end, we propose Segmental Video Reward, a mechanism to provide reward signals based on segmental sparse sampling, and Temporally Attenuated Reward, a method that mitigates temporal modeling degradation during fine-tuning. 
> Extensive experiments, both qualitative and quantitative, validate the practicality and efficacy of using image reward models in InstructVideo, significantly enhancing the visual quality of generated videos without compromising generalization capabilities. 
> Code and models will be made publicly available at this repo.


## Todo list
Note that if you can not get access to the links provided below, try using another browser or contact me by e-mail or raise an issue. 
- [ ] 🕘 Release code for fine-tuning and inference.
- [ ] 🕘 Release pre-training and fine-tuning data list (should be obtained from WebVid10M). 
- [ ] 🕘 Release pre-training and fine-tuned checkpoints.  


## Configrue the Environment
Please refer to the main [README](https://github.com/damo-vilab/i2vgen-xl/blob/main/README.MD) to configure the environment.


## Fine-tuning and Inference
Pre-trained models and details on InstrcutVideo fine-tuning and inference are coming soon. Stay tuned!

## Citation
```bibtex
@article{2023InstructVideo,
    title={InstructVideo: Instructing Video Diffusion Models with Human Feedback},
    author={Yuan, Hangjie and Zhang, Shiwei and Wang, Xiang and Wei, Yujie and Feng, Tao and Pan, Yining and Zhang, Yingya and Liu, Ziwei and Albanie, Samuel and Ni, Dong},
    booktitle={arXiv preprint arXiv:2312.12490},
    year={2023}
}
```
