import os
import sys
import torch
from einops import rearrange
import time
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import clip
from PIL import Image
import argparse
import cv2
import copy
from dino.vision_transformer import vit_small


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def calculate_metric(videos_path, prompt, target_img_files, clip_model, clip_preprocess, dino_model, dino_preprocess, chunk_size=1):

    video_capture = cv2.VideoCapture(videos_path)
    video_data_all = []
    while(True):
        ret, frame = video_capture.read()
        if (not ret) or (frame is None): break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        video_data_all.append(frame)
    
    video_data_all = torch.stack(video_data_all)
    video_data_all = video_data_all.unsqueeze(0)

    prompts = [prompt]

    target_img_list = []
    for target_img_file in target_img_files:
        image = Image.open(target_img_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        target_img_list.append(image) 

    videos_size = video_data_all.size(0)
    prompts_size = len(prompts)
    assert videos_size == prompts_size, f'videos data size {videos_size} and prompts size {prompts_size} are not equal!'


    video_data_all = rearrange(video_data_all, 'n f h w c -> n c f h w')

    video_data_list = torch.chunk(video_data_all, video_data_all.shape[0] // chunk_size, dim=0)
    prompts_list = chunks(prompts, video_data_list[0].size(0))
    CLIP_T_scores = []
    TemporalConsistency_scores= []
    CLIP_I_scores = []
    DINO_I_scores = []
    transform = T.ToPILImage()
    with torch.no_grad():
        for prompts_part, video_data_part in tqdm(zip(prompts_list, video_data_list), total=len(video_data_list)):
            # text feature
            text = clip.tokenize(prompts_part).cuda()

            t_feature = clip_model.encode_text(text)
            t_feature = F.normalize(t_feature, p=2, dim=1)

            # clip image feature
            n,c,f,h,w = video_data_part.shape
            video_data_part = rearrange(video_data_part.cuda(), 'n c f h w ->(n f) c h w')
            images = [clip_preprocess(transform(img)).unsqueeze(0) for img in video_data_part]
            images = torch.cat(images).cuda()

            gen_feature = clip_model.encode_image(images)
            gen_feature = F.normalize(gen_feature, p=2, dim=1)
            gen_feature = rearrange(gen_feature, '(n f) c -> n f c', f=f)

            if len(target_img_list) > 0:
                target_images = [clip_preprocess(target_img).unsqueeze(0) for target_img in target_img_list]
                target_images = torch.cat(target_images).cuda()
            
            # dino image feature
            images_dino = [dino_preprocess(transform(img)).unsqueeze(0) for img in video_data_part]
            images_dino = torch.cat(images_dino).cuda()
            gen_dino_feature = dino_model(images_dino)
            gen_dino_feature = F.normalize(gen_dino_feature, p=2, dim=1)
            gen_dino_feature = rearrange(gen_dino_feature, '(n f) c -> n f c', f=f)

            if len(target_img_list) > 0:
                target_images_dino = [dino_preprocess(target_img).unsqueeze(0) for target_img in target_img_list]
                target_images_dino = torch.cat(target_images_dino).cuda()
            
            if len(target_img_list) > 0:
                target_feature = clip_model.encode_image(target_images)
                target_feature = F.normalize(target_feature, p=2, dim=1)

                target_dino_feature = dino_model(target_images_dino)
                target_dino_feature = F.normalize(target_dino_feature, p=2, dim=1)

            # calculate similarity
            gen_feature_copy = copy.deepcopy(gen_feature)
            gen_feature = gen_feature.mean(dim=1)
            gen_dino_feature = gen_dino_feature.mean(dim=1)
            gen_feature_shift = copy.deepcopy(gen_feature_copy[:, 1:, ...])
            CLIP_T = (t_feature * gen_feature).sum(dim=1).cpu()

            if len(target_img_list) > 0:
                img_similar_list = []
                dino_img_similar_list = []
                for i in range(target_feature.shape[0]):
                    img_similar = (target_feature[i] * gen_feature).sum(dim=1).cpu()
                    img_similar_list.append(img_similar)

                    dino_img_similar = (target_dino_feature[i] * gen_dino_feature).sum(dim=1).cpu()
                    dino_img_similar_list.append(dino_img_similar)

                img_similar_scores = torch.cat(img_similar_list)
                CLIP_I = img_similar_scores.mean()
                CLIP_I_scores.append(CLIP_I)

                dino_img_similar_scores = torch.cat(dino_img_similar_list)
                DINO_I = dino_img_similar_scores.mean()
                DINO_I_scores.append(DINO_I)

            TemporalConsistency = (gen_feature_copy[:, :-1, ...] * gen_feature_shift).sum(dim=2).cpu()
            CLIP_T_scores.append(CLIP_T)
            TemporalConsistency_scores.append(TemporalConsistency)

    CLIP_T_scores = torch.cat(CLIP_T_scores)
    TemporalConsistency_scores = torch.cat(TemporalConsistency_scores)
    CLIP_T_mean_score = CLIP_T_scores.mean().cpu().item()
    TemporalConsistency_mean_score = TemporalConsistency_scores.mean().cpu().item()

    if len(target_img_list) > 0:
        CLIP_I_scores = torch.stack(CLIP_I_scores)
        DINO_I_scores = torch.stack(DINO_I_scores)
        CLIP_I_mean_score = CLIP_I_scores.mean().cpu().item()
        DINO_I_mean_score = DINO_I_scores.mean().cpu().item()
    else:
        CLIP_I_mean_score = 0
        DINO_I_mean_score = 0
    print(f'CLIP-T: {CLIP_T_mean_score:.4f}, CLIP-I: {CLIP_I_mean_score:.4f}, DINO-I: {DINO_I_mean_score:.4f}, TemporalConsistency: {TemporalConsistency_mean_score:.4f}')
    return CLIP_T_mean_score, CLIP_I_mean_score, DINO_I_mean_score, TemporalConsistency_mean_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Important args')
    parser.add_argument("--videos_dir_path", type=str,
                        help="")
    parser.add_argument("--prompts_path", type=str, 
                        help="path of the prompts file")
    args = parser.parse_args()

    videos_dir_path = args.videos_dir_path
    prompts_path = args.prompts_path

    # load model
    # [model] clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
    # [model] dino
    dino_vits16 = vit_small()
    dino_vits16_path = "/path/to/dino/dino_deitsmall16_pretrain.pth"  # https://github.com/facebookresearch/dino
    dino_vits16.load_state_dict(torch.load(dino_vits16_path, map_location='cpu'))
    dino_vits16 = dino_vits16.cuda()

    dino_preprocess = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    video_files = [os.path.join(videos_dir_path, f) for f in os.listdir(videos_dir_path) if f.endswith('.mp4') or f.endswith('.gif')]
    
    prompts_dict = {}
    with open(prompts_path, 'r', encoding='utf-8') as prompts_file:
        for line in prompts_file:
            parts = line.strip().split("|||")
            video_name = parts[0]
            prompts_dict[video_name] = (parts[1], parts[2])

    print(f"-----------------test {len(video_files)} videos for video customization!---------------")
    if len(video_files) != len(prompts_dict):
        print(f"The number of videos {len(video_files)} and text prompts {len(prompts_dict)} are inconsistent!!")
        exit()

    CLIP_T_score_list = []
    CLIP_I_score_list = []
    DINO_I_score_list = []
    TemporalConsistency_score_list = []
    for idx, video in enumerate(video_files):
        video_name = video.split('/')[-1]
        image_dir, prompt = prompts_dict[video_name]

        image_name = image_dir.split('/')[-1]
        target_img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        print(f'test {idx}th video {video_name}---{image_name}---{prompt}')
        CLIP_T_score, CLIP_I_score, DINO_I_score, TemporalConsistency_score = calculate_metric(video, prompt, target_img_files, clip_model, clip_preprocess, dino_vits16, dino_preprocess)
        CLIP_T_score_list.append(CLIP_T_score)
        CLIP_I_score_list.append(CLIP_I_score)
        DINO_I_score_list.append(DINO_I_score)
        TemporalConsistency_score_list.append(TemporalConsistency_score)

    CLIP_T_mean = sum(CLIP_T_score_list) / len(CLIP_T_score_list)
    CLIP_I_mean = sum(CLIP_I_score_list) / len(CLIP_I_score_list)
    DINO_I_mean = sum(DINO_I_score_list) / len(DINO_I_score_list)
    TemporalConsistency_mean = sum(TemporalConsistency_score_list) / len(TemporalConsistency_score_list)
    # print(f'CLIP-T_score_list: {CLIP_T_score_list}')
    # print(f'CLIP_I_score_list: {CLIP_I_score_list}')
    # print(f'DINO_I_score_list: {DINO_I_score_list}')
    # print(f'TemporalConsistency_score_list: {TemporalConsistency_score_list}')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f'Final CLIP-T: {CLIP_T_mean:.4f}, CLIP-I: {CLIP_I_mean:.4f}, DINO-I: {DINO_I_mean:.4f},  TemporalConsistency: {TemporalConsistency_mean:.4f}')
