# Metrics calculation of DreamVideo

The code includes the calculation of metrics for **CLIP-T**, **CLIP-I**, **DINO-I**, and **Temporal Consistency**.

## Setup

1. Prepare environment.
 ```
 pip install git+https://github.com/openai/CLIP.git
 ```

2. Prepare your videos and test_prompts as in `metric/examples`.
The format of each line of the `test_prompts.txt` is as follows:
```
{video_filename}|||{reference_img_folder}|||{text_prompt}
```

3. Replace `dino_vits16_path` in cal_metric_DreamVideo.py. See [DINO Repo](https://github.com/facebookresearch/dino).

## Example

Run the following command to calculate metrics:
```
python metric/cal_metric_DreamVideo.py --videos_dir_path metric/examples/videos --prompts_path metric/examples/test_prompts.txt
```

## BibTeX

If this code is useful to you, please cite our corresponding paper.


```bibtex
@inproceedings{wei2023dreamvideo,
  title={DreamVideo: Composing Your Dream Videos with Customized Subject and Motion},
  author={Wei, Yujie and Zhang, Shiwei and Qing, Zhiwu and Yuan, Hangjie and Liu, Zhiheng and Liu, Yu and Zhang, Yingya and Zhou, Jingren and Shan, Hongming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6537--6549},
  year={2024}
}
```