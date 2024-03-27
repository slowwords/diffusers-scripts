import torch
from diffusers import StableDiffusionXLInpaintPipeline
from omegaconf import OmegaConf
from io import BytesIO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
args = parser.parse_args()

pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
    pretrained_model_link_or_path=args.model_path,
    original_config_file=args.config_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(args.device)
pipeline.save_pretrained(
    args.out_path,
    variant='fp16',
)

