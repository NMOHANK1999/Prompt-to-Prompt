import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import importlib
import prompt2prompt
importlib.reload(prompt2prompt)
from prompt2prompt import MyLDMPipeline, MySharedAttentionSwapper, unet_inject_attention_modules, create_image_grid

prompt = [
        "A painting of a squirrel eating a burger",
        "A painting of a cat eating a burger",
        "A painting of a lion eating a burger",
        "A painting of a deer eating a burger",
    ]

num_inference_steps = 50
guidance_scale = 7.5
on_colab = False


pipe = MyLDMPipeline(num_inference_steps, guidance_scale)
swapper = MySharedAttentionSwapper(prompt, pipe.tokenizer, prop_steps_cross=0.0, prop_steps_self=0.0)
unet_inject_attention_modules(pipe.unet, swapper)
image = pipe._generate_image_from_text(prompt, pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.scheduler, pipe.feature_extractor, pipe.safety_checker, swapper, pipe.num_inference_steps, pipe.guidance_scale, False)
grid_image = create_image_grid(image)
if not on_colab:
    grid_image.show()