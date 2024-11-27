import torchvision
import PIL
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch

import math
import cv2

to_pillow_fn = lambda x: PIL.Image.fromarray(x[0].permute(1,2,0).detach().cpu().to(torch.uint8).numpy())
class Inpainter():
    def __init__(self, model_name="runwayml/stable-diffusion-inpainting", device="cuda", use_controlnet=True):
        print("WARNING DEBUG MODE USE ONLY OPENCV")
        self.name = "cv_inpainter"
    
    def inpaint_cv(self, init_image_tensor: torch.Tensor, mask_image_tensor: torch.Tensor,inpaintRadius=64):
        init_image = to_pillow_fn(init_image_tensor)
        origin_size = init_image.size
        if mask_image_tensor.shape[1] == 1:
            mask_image_tensor = mask_image_tensor.repeat(1,3,1,1)
        if mask_image_tensor.max() <=1.1: # if mask_image_tensor value is [0,1] , update to [0, 255]
            mask_image_tensor = mask_image_tensor * 255
            mask_image_tensor = mask_image_tensor.clamp(0, 255)
        mask_image = to_pillow_fn(mask_image_tensor).convert('L')
        img = np.array(init_image)
        mask = np.array(mask_image)
        dst_TELEA = cv2.inpaint(img,mask,inpaintRadius,cv2.INPAINT_TELEA)
        dst_TELEA = Image.fromarray(dst_TELEA)
        inpaint_img  = torchvision.transforms.functional.pil_to_tensor(dst_TELEA).unsqueeze(0).to(init_image_tensor.device)
        return inpaint_img

    @torch.no_grad()
    def inpaint(self, init_image_tensor: torch.Tensor, mask_image_tensor: torch.Tensor, control_image_tensor = None, prompt="", resize_to_area_limit_before_inpaint=False):
        result = self.inpaint_cv(init_image_tensor, mask_image_tensor)
        
        return result
    
inpainter = Inpainter()
