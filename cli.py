import os
import argparse
import math
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection
import torch
import torch.nn as nn
from src.pose_guider import PoseGuider
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
import random
import bitsandbytes as bnb
import gradio as gr

example_path = os.path.join(os.path.dirname(__file__), 'example')

# 8-bit quantization function
def quantize_model_to_8bit(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized_linear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0
            )
            quantized_linear.weight = bnb.nn.Int8Params(module.weight.data, requires_grad=False)
            if module.bias is not None:
                quantized_linear.bias = bnb.nn.Int8Params(module.bias.detach().clone(), requires_grad=False)
            # quantized_linear.to(module.device)
            setattr(model, name, quantized_linear)
        else:
            quantize_model_to_8bit(module)

# 4-bit quantization function
def quantize_model_to_4bit(model, compute_dtype=torch.bfloat16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 현대적인 bitsandbytes API 사용
            quantized_linear = bnb.modules.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
            )
            # 가중치 초기화
            quantized_linear.load_state_dict(module.state_dict())
            setattr(model, name, quantized_linear)
        else:
            quantize_model_to_4bit(module, compute_dtype=compute_dtype)

# FitDiTGenerator class with quantization support
class FitDiTGenerator:
    def __init__(self, model_root, offload=False, aggressive_offload=False, device="cuda:0", with_fp16=False):
        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        self.transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
            os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype
        )
        self.transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
            os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype
        )
        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512)
        )
        self.pose_guider.load_state_dict(
            torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin"))
        )
        self.image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=weight_dtype
        )
        self.image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype
        )

        self.pose_guider.to(device=device, dtype=weight_dtype)
        self.image_encoder_large.to(device=device)
        self.image_encoder_bigG.to(device=device)

        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            model_root,
            torch_dtype=weight_dtype,
            transformer_garm=self.transformer_garm,
            transformer_vton=self.transformer_vton,
            pose_guider=self.pose_guider,
            image_encoder_large=self.image_encoder_large,
            image_encoder_bigG=self.image_encoder_bigG
        )

        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.pipeline.to(device)
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device)
            self.parsing_model = Parsing(model_root=model_root, device=device)

    # Quantization method
    def quantize(self, dtype):
        if dtype == "q8":
            quantize_model_to_8bit(self.transformer_garm)
            quantize_model_to_8bit(self.transformer_vton)
            # quantize_model_to_8bit(self.pose_guider)
            quantize_model_to_8bit(self.image_encoder_large)
            quantize_model_to_8bit(self.image_encoder_bigG)
        elif dtype == "q4":
            quantize_model_to_4bit(self.transformer_garm, compute_dtype=torch.bfloat16)
            quantize_model_to_4bit(self.transformer_vton, compute_dtype=torch.bfloat16)
            # quantize_model_to_4bit(self.pose_guider, compute_dtype=torch.bfloat16)
            quantize_model_to_4bit(self.image_encoder_large, compute_dtype=torch.bfloat16)
            quantize_model_to_4bit(self.image_encoder_bigG, compute_dtype=torch.bfloat16)

    def generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img = Image.open(vton_img)
            vton_img_det = resize_image(vton_img)
            pose_image, _, _, candidate = self.dwprocessor(np.array(vton_img_det)[:, :, ::-1])
            candidate[candidate < 0] = 0
            candidate = candidate[0]
            candidate[:, 0] *= vton_img_det.width
            candidate[:, 1] *= vton_img_det.height

            pose_image = pose_image[:, :, ::-1]  # Convert to RGB
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {
                'background': np.array(vton_img.convert("RGBA")),
                'layers': [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:, :, np.newaxis]), axis=2)],
                'composite': np.array(masked_vton_img.convert("RGBA"))
            }
            return im, pose_image

    def process(self, vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution):
        assert resolution in ["768x1024", "1152x1536", "1536x2048"]
        new_width, new_height = map(int, resolution.split("x"))
        with torch.inference_mode():
            garm_img = Image.open(garm_img)
            vton_img = Image.open(vton_img)

            model_image_size = vton_img.size
            garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

            mask = pre_mask["layers"][0][:, :, 3]
            mask = Image.fromarray(mask)
            mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0, 0, 0))
            mask = mask.convert("L")
            pose_image = Image.fromarray(np.array(pose_image))
            pose_image, _, _ = pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0, 0, 0))

            if seed == -1:
                seed = random.randint(0, 2147483647)
            res = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images_per_prompt
            ).images
            for idx in range(len(res)):
                res[idx] = unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
            return res

# Utility functions
def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    new_size = (new_width, round(old_height * ratio_w)) if ratio_w < ratio_h else (round(old_width * ratio_h), new_height)
    im_resized = im.resize(new_size, mode)
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)
    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))
    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    cropped_im = padded_im.crop((pad_w, pad_h, width - pad_w, height - pad_h))
    return cropped_im.resize((original_width, original_height), Image.LANCZOS)

def resize_image(img, target_size=768):
    width, height = img.size
    scale = target_size / min(width, height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    return img.resize((new_width, new_height), Image.LANCZOS)

# Gradio demo creation function
HEADER = """
<h1 style="text-align: center;"> FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on </h1>
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://github.com/BoyuanJiang/FitDiT" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://arxiv.org/abs/2411.10499" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href="http://demo.fitdit.byjiang.com/" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://byjiang.com/FitDiT/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://raw.githubusercontent.com/BoyuanJiang/FitDiT/refs/heads/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>
<br>
FitDiT is designed for high-fidelity virtual try-on using Diffusion Transformers (DiT). It can only be used for <b>Non-commercial Use</b>.<br>
If you like our work, please star <a href="https://github.com/BoyuanJiang/FitDiT" style="color: blue; text-decoration: underline;">our github repository</a>.
"""

def create_demo(model_path, device, offload, aggressive_offload, with_fp16, dtype):
    generator = FitDiTGenerator(model_path, offload, aggressive_offload, device, with_fp16)
    generator.quantize(dtype)  # Apply quantization based on dtype argument
    
    with gr.Blocks(title="FitDiT") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column():
                vton_img = gr.Image(label="Model", sources=None, type="filepath", height=512)
            with gr.Column():
                garm_img = gr.Image(label="Garment", sources=None, type="filepath", height=512)
        with gr.Row():
            with gr.Column():
                masked_vton_img = gr.ImageEditor(label="masked_vton_img", type="numpy", height=512, interactive=True, brush=gr.Brush(default_color="rgb(127, 127, 127)", colors=["rgb(128, 128, 128)"]))
                pose_image = gr.Image(label="pose_image", visible=False, interactive=False)
            with gr.Column():
                result_gallery = gr.Gallery(label="Output", elem_id="output-img", interactive=False, columns=[2], rows=[2], object_fit="contain", height="auto")
        with gr.Row():
            with gr.Column():
                offset_top = gr.Slider(label="mask offset top", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_bottom = gr.Slider(label="mask offset bottom", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_left = gr.Slider(label="mask offset left", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_right = gr.Slider(label="mask offset right", minimum=-200, maximum=200, step=1, value=0)
        with gr.Row():
            with gr.Column():
                n_steps = gr.Slider(label="Steps", minimum=15, maximum=30, value=20, step=1)
            with gr.Column():
                image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2, step=0.1)
            with gr.Column():
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
            with gr.Column():
                num_images_per_prompt = gr.Slider(label="num_images", minimum=1, maximum=4, step=1, value=1)
        with gr.Row():
            with gr.Column():
                example = gr.Examples(
                    label="Model (upper-body)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/0279.jpg'),
                        os.path.join(example_path, 'model/0303.jpg'),
                        os.path.join(example_path, 'model/2.jpg'),
                        os.path.join(example_path, 'model/0083.jpg'),
                    ])
                example = gr.Examples(
                    label="Model (upper-body/lower-body)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/0.jpg'),
                        os.path.join(example_path, 'model/0179.jpg'),
                        os.path.join(example_path, 'model/0223.jpg'),
                        os.path.join(example_path, 'model/0347.jpg'),
                    ])
                example = gr.Examples(
                    label="Model (dresses)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/4.jpg'),
                        os.path.join(example_path, 'model/5.jpg'),
                        os.path.join(example_path, 'model/6.jpg'),
                        os.path.join(example_path, 'model/7.jpg'),
                    ])
            with gr.Column():
                example = gr.Examples(
                    label="Garment (upper-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/12.png'),
                        os.path.join(example_path, 'garment/0012.jpg'),
                        os.path.join(example_path, 'garment/0047.jpg'),
                        os.path.join(example_path, 'garment/0049.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (lower-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/0317.jpg'),
                        os.path.join(example_path, 'garment/0327.jpg'),
                        os.path.join(example_path, 'garment/0329.jpg'),
                        os.path.join(example_path, 'garment/0362.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (dresses)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/8.jpg'),
                        os.path.join(example_path, 'garment/9.png'),
                        os.path.join(example_path, 'garment/10.jpg'),
                        os.path.join(example_path, 'garment/11.jpg'),
                    ])
            with gr.Column():
                category = gr.Dropdown(label="Garment category", choices=["Upper-body", "Lower-body", "Dresses"], value="Upper-body")
                resolution = gr.Dropdown(label="Try-on resolution", choices=["768x1024", "1152x1536", "1536x2048"], value="1152x1536")
            with gr.Column():
                run_mask_button = gr.Button(value="Step1: Run Mask")
                run_button = gr.Button(value="Step2: Run Try-on")

        ips1 = [vton_img, category, offset_top, offset_bottom, offset_left, offset_right]
        ips2 = [vton_img, garm_img, masked_vton_img, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution]
        run_mask_button.click(fn=generator.generate_mask, inputs=ips1, outputs=[masked_vton_img, pose_image])
        run_button.click(fn=generator.process, inputs=ips2, outputs=[result_gallery])
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitDiT Gradio Demo with Quantization")
    parser.add_argument("--model_path", type=str, required=True, help="The path of FitDiT model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--dtype", default="q8", choices=["q8", "q4"], help="Data type for quantization (q8 or q4)")
    parser.add_argument("--fp16", action="store_true", help="Load model with fp16, default is bf16")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use.")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use.")
    args = parser.parse_args()
    
    demo = create_demo(args.model_path, args.device, args.offload, args.aggressive_offload, args.fp16, args.dtype)
    demo.launch(share=True)