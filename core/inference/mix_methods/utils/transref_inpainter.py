import sys

import torch

sys.path.append('core/inference/mix_methods/utils/TransRef/')
from options.test_options import TestOptions
from models.model import create_model
sys.path.remove('core/inference/mix_methods/utils/TransRef/')
import PIL
to_pillow_fn = lambda x: PIL.Image.fromarray(x[0].permute(1,2,0).detach().cpu().to(torch.uint8).numpy())
import torchvision.transforms as transforms
import os

class Inpainter():
    def __init__(self, model_name="TransRef", device="cuda", use_controlnet=True):
        self.name = "transref_inpainter"
        self.device = device
        
        self.network_pth = os.path.join(os.path.dirname(__file__),"TransRef","400_Trans.pth")
        print(f'Loading networks from: {self.network_pth}')

        opt = TestOptions().get_test_option_cfg()
        
        model = create_model(opt)
        TransRef = torch.load(self.network_pth)
        model.model.load_state_dict(TransRef['net'], strict=False)
        model.model.to(device)
        self.model = model

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        

    @torch.no_grad()
    def inpaint(self, init_image_tensor: torch.Tensor, mask_image_tensor: torch.Tensor, control_image_tensor = None, prompt="", resize_to_area_limit_before_inpaint=False):


        origin_device = init_image_tensor.device
        origin_h, origin_w = init_image_tensor.shape[2],init_image_tensor.shape[3]  
     
        new_h, new_w = 512, 512

        init_image = to_pillow_fn(init_image_tensor)
        ref_image = to_pillow_fn(control_image_tensor)

        detail = self.img_transform(init_image).unsqueeze(0).to(self.device)
        # resize to new_h, new_w
        detail =  torch.nn.functional.interpolate(detail, size=[new_h, new_w], mode='bilinear')
        input_mask = mask_image_tensor.to(self.device)
        input_mask = torch.nn.functional.interpolate(input_mask, size=[new_h, new_w], mode='bilinear')
        reference = self.img_transform(ref_image).unsqueeze(0).to(self.device)
        reference =  torch.nn.functional.interpolate(reference, size=[new_h, new_w], mode='bilinear')


        self.model.set_input(detail,input_mask,reference)
        self.model.forward()

        fake_out = self.model.out

        fake_out = fake_out * input_mask + detail*(1-input_mask)
        
        # resize back
        output = torch.nn.functional.interpolate(fake_out, size=[origin_h, origin_w], mode='bilinear')
        output = (output * 127.5 + 127.5).round().clamp(0, 255)
        output = output.to(device = origin_device, dtype = torch.uint8)
  

        return output
    
inpainter = Inpainter()
