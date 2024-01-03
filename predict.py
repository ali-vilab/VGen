# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from subprocess import call
import shutil
import yaml
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Input prompt."),
        max_frames: int = Input(
            description="Number of frames in the output", default=16
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        infer_yaml = "infer.yaml"
        test_file_name = "test_prompt_image"
        with open(f"{test_file_name}.txt", "w") as file:
            file.write(f"{str(image)}|||{prompt}")

        with open("configs/i2vgen_xl_infer.yaml", "r") as file:
            config = yaml.safe_load(file)

        updated_config = {
            "max_frames": max_frames,
            "seed": seed,
            "test_list_path": f"{test_file_name}.txt",
            "guide_scale": guidance_scale,
            "round": 1,
        }

        for key, value in updated_config.items():
            if key in config:
                config[key] = value

        with open(infer_yaml, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        infer = (
            "python inference.py --cfg "
            + infer_yaml
            + " test_list_path "
            + f"{test_file_name}.txt"
            + " test_model models/i2vgen_xl_00854500.pth"
        )

        default_output_dir = f"workspace/experiments/{test_file_name}/"
        if os.path.exists(default_output_dir):
            shutil.rmtree(default_output_dir)

        call(infer, shell=True)
        out_path = "/tmp/out.mp4"

        files = os.listdir(default_output_dir)
        print(f"Files saved: {files}")
        mp4_file = next((file for file in files if file.endswith(".mp4")), None)

        assert mp4_file is not None, "No output mp4 file found"
        shutil.copy(os.path.join(default_output_dir, mp4_file), out_path)
        return Path(out_path)
