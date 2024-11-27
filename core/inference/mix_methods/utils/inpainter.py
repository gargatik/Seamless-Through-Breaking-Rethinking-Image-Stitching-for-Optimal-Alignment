import torchvision
import PIL
from PIL import ImageOps
import numpy as np
import torch

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, AutoPipelineForInpainting
import math

to_pillow_fn = lambda x: PIL.Image.fromarray(x[0].permute(1,2,0).detach().cpu().to(torch.uint8).numpy())
class Inpainter():
    def __init__(self, model_name="runwayml/stable-diffusion-inpainting", device="cuda", use_controlnet=True):
        self.device = device
        self.name = "inpainter"
        if use_controlnet:
            self.controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")
            self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                model_name, controlnet=self.controlnet, torch_dtype=torch.float16, variant="fp16",
            )
        else:
             self.pipeline = AutoPipelineForInpainting.from_pretrained(
                model_name, torch_dtype=torch.float16, variant="fp16"
            )
        self.pipeline = self.pipeline.to(device)

        
    def make_inpaint_condition(self, init_image, mask_image):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    
        assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)

        return init_image
    def apply_overlay(
        self,
        mask: PIL.Image.Image,
        init_image: PIL.Image.Image,
        image: PIL.Image.Image,
        crop_coords=None #Optional[Tuple[int, int, int, int]] = None,
    ) -> PIL.Image.Image:
        """
        overlay the inpaint output to the original image
        """

        width, height = image.width, image.height
        origin_size = image.size

        init_image = init_image.resize(origin_size)
        mask = mask.resize(origin_size)

        init_image_masked = PIL.Image.new("RGBa", (width, height))
        init_image_masked.paste(init_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))
        init_image_masked = init_image_masked.convert("RGBA")

        if crop_coords is not None:
            x, y, x2, y2 = crop_coords
            w = x2 - x
            h = y2 - y
            base_image = PIL.Image.new("RGBA", (width, height))
            image = self.resize(image, height=h, width=w, resize_mode="crop")
            base_image.paste(image, (x, y))
            image = base_image.convert("RGB")

        image = image.convert("RGBA")
        image.alpha_composite(init_image_masked)
        image = image.convert("RGB")

        return image
    
    @torch.no_grad()
    def inpaint(self, init_image_tensor: torch.Tensor, mask_image_tensor: torch.Tensor, control_image_tensor = None, prompt="", resize_to_area_limit_before_inpaint=False):
        
        init_image = to_pillow_fn(init_image_tensor)
        origin_size = init_image.size
        if mask_image_tensor.shape[1] == 1:
            mask_image_tensor = mask_image_tensor.repeat(1,3,1,1)
        if mask_image_tensor.max() <=1.1: # if mask_image_tensor value is [0,1] , update to [0, 255]
            mask_image_tensor = mask_image_tensor * 255
            mask_image_tensor = mask_image_tensor.clamp(0, 255)
        mask_image = to_pillow_fn(mask_image_tensor)
        
        if resize_to_area_limit_before_inpaint:
            # max_size = max(origin_size)
            area_factor = resize_to_area_limit_before_inpaint / (origin_size[0] * origin_size[1])
            scale_factor = math.sqrt(area_factor)
            target_size = (origin_size[0] * scale_factor, origin_size[1]* scale_factor)
            target_size = (int(target_size[0]), int(target_size[1]))
            init_image = init_image.resize(target_size)
            mask_image = mask_image.resize(target_size)

        if control_image_tensor is None:
            control_image_tensor = self.make_inpaint_condition(init_image, mask_image)
        result = self.pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image_tensor, \
                              # strength = 0.6, \
                              # guidance_scale=2.5, \
                              ).images[0]
        result = result.resize(origin_size)
        
        result  = torchvision.transforms.functional.pil_to_tensor(result).unsqueeze(0).to(init_image_tensor.device)
        
        
        return result
    
inpainter = Inpainter()
