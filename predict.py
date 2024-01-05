# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import yaml
import pynvml
from PIL import Image
import torch.distributed as dist
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange
from cog import BasePredictor, Input, Path

from tools.modules.config import cfg
from utils.multi_port import find_free_port
from utils.seed import setup_seed
from utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from utils.assign_cfg import assign_signle_cfg
from utils.registry_class import MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION
import utils.transforms as data


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        with open("configs/i2vgen_xl_infer.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.cfg = assign_signle_cfg(cfg, config, "vldm_cfg")

        for k, v in config.items():
            if isinstance(v, dict) and k in self.cfg:
                self.cfg[k].update(v)
            else:
                self.cfg[k] = v

        if not "MASTER_ADDR" in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = find_free_port()

        self.cfg.gpu = 0
        self.cfg.pmi_rank = int(os.getenv("RANK", 0))
        self.cfg.pmi_world_size = int(os.getenv("WORLD_SIZE", 1))
        self.cfg.gpus_per_machine = torch.cuda.device_count()
        self.cfg.world_size = self.cfg.pmi_world_size * self.cfg.gpus_per_machine

        torch.cuda.set_device(self.cfg.gpu)
        torch.backends.cudnn.benchmark = True

        self.cfg.rank = self.cfg.pmi_rank * self.cfg.gpus_per_machine + self.cfg.gpu
        dist.init_process_group(
            backend="nccl", world_size=self.cfg.world_size, rank=self.cfg.rank
        )

        # [Diffusion]
        self.diffusion = DIFFUSION.build(self.cfg.Diffusion)

        # [Model] embedder
        self.clip_encoder = EMBEDDER.build(self.cfg.embedder)
        self.clip_encoder.model.to(self.cfg.gpu)
        _, _, zero_y_negative = self.clip_encoder(text=self.cfg.negative_prompt)
        self.zero_y_negative = zero_y_negative.detach()
        self.black_image_feature = torch.zeros([1, 1, self.cfg.UNet.y_dim]).cuda()

        # [Model] auotoencoder
        self.autoencoder = AUTO_ENCODER.build(self.cfg.auto_encoder)
        self.autoencoder.eval()  # freeze
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.cuda()

        # [Model] UNet
        self.model = MODEL.build(self.cfg.UNet)
        checkpoint_dict = torch.load(self.cfg.test_model, map_location="cpu")
        state_dict = checkpoint_dict["state_dict"]
        status = self.model.load_state_dict(state_dict, strict=True)
        print("Load model from {} with status {}".format(self.cfg.test_model, status))
        self.model = self.model.to(self.cfg.gpu)
        self.model.eval()
        self.model = DistributedDataParallel(self.model, device_ids=[self.cfg.gpu])
        torch.cuda.empty_cache()

        print("Models loaded!")

    def predict(
        self,
        image: Path = Input(description="Input image."),
        prompt: str = Input(description="Describe the input image."),
        max_frames: int = Input(
            description="Number of frames in the output", default=16, ge=2
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        setup_seed(seed)

        # [Data] Data Transform
        train_trans = data.Compose(
            [
                data.CenterCropWide(size=self.cfg.resolution),
                data.ToTensor(),
                data.Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )

        vit_trans = data.Compose(
            [
                data.CenterCropWide(
                    size=(self.cfg.resolution[0], self.cfg.resolution[0])
                ),
                data.Resize(self.cfg.vit_resolution),
                data.ToTensor(),
                data.Normalize(mean=self.cfg.vit_mean, std=self.cfg.vit_std),
            ]
        )
        captions = [prompt]
        with torch.no_grad():
            image_tensor = vit_trans(image)
            image_tensor = image_tensor.unsqueeze(0)
            y_visual, y_text, y_words = self.clip_encoder(
                image=image_tensor, text=captions
            )
            y_visual = y_visual.unsqueeze(1)

        fps_tensor = torch.tensor(
            [self.cfg.target_fps], dtype=torch.long, device=self.cfg.gpu
        )
        image_id_tensor = train_trans([image]).to(self.cfg.gpu)
        local_image = self.autoencoder.encode_firsr_stage(
            image_id_tensor, self.cfg.scale_factor
        ).detach()
        local_image = local_image.unsqueeze(2).repeat_interleave(
            repeats=max_frames, dim=2
        )

        with torch.no_grad():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB")
            # Sample images
            with amp.autocast(enabled=self.cfg.use_fp16):
                noise = torch.randn(
                    [
                        1,
                        4,
                        max_frames,
                        int(self.cfg.resolution[1] / self.cfg.scale),
                        int(self.cfg.resolution[0] / self.cfg.scale),
                    ]
                )
                noise = noise.to(self.cfg.gpu)

                infer_img = (
                    self.black_image_feature if self.cfg.use_zero_infer else None
                )
                model_kwargs = [
                    {
                        "y": y_words,
                        "image": y_visual,
                        "local_image": local_image,
                        "fps": fps_tensor,
                    },
                    {
                        "y": self.zero_y_negative,
                        "image": infer_img,
                        "local_image": local_image,
                        "fps": fps_tensor,
                    },
                ]
                video_data = self.diffusion.ddim_sample_loop(
                    noise=noise,
                    model=self.model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=guidance_scale,
                    ddim_timesteps=num_inference_steps,
                    eta=0.0,
                )

        video_data = 1.0 / self.cfg.scale_factor * video_data  # [1, 4, 32, 46]
        video_data = rearrange(video_data, "b c f h w -> (b f) c h w")
        chunk_size = min(self.cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(
            video_data, video_data.shape[0] // chunk_size, dim=0
        )
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = self.autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(
            video_data, "(b f) c h w -> b c f h w", b=self.cfg.batch_size
        )

        text_size = cfg.resolution[-1]
        out_path = "/tmp/out.mp4"
        try:
            save_i2vgen_video_safe(
                out_path,
                video_data.cpu(),
                captions,
                self.cfg.mean,
                self.cfg.std,
                text_size,
            )
        except Exception as e:
            print(f"Step: save text or video error with {e}")

        torch.cuda.synchronize()
        dist.barrier()

        return Path(out_path)
