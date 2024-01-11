import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .attention_processor_total import LoRAAttnProcessor, LoRAIPAttnProcessor
    
from .resampler import PerceiverAttention, FeedForward, Resampler

class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)



class ProjPlusModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class IPAdapterFaceIDPlus:
    def __init__(self, sd_pipe, image_encoder_path_faceid, image_encoder_path_normal, ip_ckpt, device, 
                 lora_rank=128, num_tokens_faceid=4,  num_tokens_normal=16, torch_dtype=torch.float16):
        self.device = device
        self.image_encoder_path_faceid = image_encoder_path_faceid
        self.image_encoder_path_normal = image_encoder_path_normal
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens_faceid = num_tokens_faceid
        self.num_tokens_normal = num_tokens_normal
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder_faceid = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path_faceid).to(
            self.device, dtype=self.torch_dtype
        )
        self.image_encoder_normal = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path_normal).to(
            self.device, dtype=self.torch_dtype
        )
        
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model_faceid, self.image_proj_model_normal = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model_faceid = ProjPlusModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens_faceid,
        ).to(self.device, dtype=self.torch_dtype)
        
        image_proj_model_normal = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens_normal,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        
        return image_proj_model_faceid, image_proj_model_normal

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank=self.lora_rank,
                    num_tokens_faceid=self.num_tokens_faceid, num_tokens_normal=self.num_tokens_normal
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            # Don't support this format
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            
        self.image_proj_model_faceid.load_state_dict(state_dict["image_proj_faceid"])
        self.image_proj_model_faceid.load_state_dict(state_dict["image_proj_normal"])
        
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, faceid_embeds, face_image, template_img, s_scale, shortcut):
        
        # for faceid, faceid and normal use different clip model
        if isinstance(face_image, Image.Image):
            pil_image = [face_image]
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.image_encoder_faceid(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder_faceid(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_embeds_faceid = self.image_proj_model_faceid(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds_faceid = self.image_proj_model_faceid(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)
        
        # for normal
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder_normal(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds_normal = self.image_proj_model_normal(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder_normal(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds_normal = self.image_proj_model_normal(uncond_clip_image_embeds)
        
        # concatenate
        image_prompt_embeds = torch.cat([image_prompt_embeds_faceid, image_prompt_embeds_normal], dim=1)
        uncond_image_prompt_embeds = torch.cat([uncond_image_prompt_embeds_faceid, uncond_image_prompt_embeds_normal], dim=1)
        
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale_faceid, scale_normal):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale_faceid = scale_faceid
                attn_processor.scale_normal = scale_normal

    def generate(
        self,
        face_image=None,
        faceid_embeds=None,
        template_img=None,
        prompt=None,
        negative_prompt=None,
        scale_faceid=1.0,
        scale_normal=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=False,
        **kwargs,
    ):
        self.set_scale(scale_faceid, scale_normal)
       
        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image, template_img, s_scale, shortcut)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


