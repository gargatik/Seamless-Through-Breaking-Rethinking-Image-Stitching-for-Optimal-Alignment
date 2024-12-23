import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.model import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms

import os
# gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    results_dir = r'./result/'
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)

    opt = TestOptions().parse()
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    model = create_model(opt)

    TransRef = torch.load("/workspace/FlowFormerPlusPlus/core/inference/mix_methods/utils/TransRef/250_Trans.pth")

    # model.model.load_state_dict(TransRef['net'], ,strict=False)
    model.model.load_state_dict(TransRef['net'], strict=False)

    input_mask_paths = glob('{:s}/*'.format("/workspace/FlowFormerPlusPlus/core/inference/mix_methods/utils/TransRef/prerpocess_by_atik/inpaint_by_other_mask/"))
    input_mask_paths.sort()
    de_paths = glob('{:s}/*'.format("/workspace/FlowFormerPlusPlus/core/inference/mix_methods/utils/TransRef/prerpocess_by_atik/img_will_inpaint/"))
    de_paths.sort()
    ref_paths = glob('{:s}/*'.format("/workspace/FlowFormerPlusPlus/core/inference/mix_methods/utils/TransRef/prerpocess_by_atik/img_will_inpaint/"))
    ref_paths.sort()

    image_len = len(de_paths)
    
    all_time =0
    count = 0
    
    for i in tqdm(range(image_len)):
        
        path_im = input_mask_paths[i]
        path_de = de_paths[i]
        path_rf = ref_paths[i]
        (filepath,tempfilename) = os.path.split(path_rf)
        (filename,extension) = os.path.splitext(tempfilename)
        
        input_mask = Image.open(path_im).convert("RGB")
        detail = Image.open(path_de)
        reference = Image.open(path_rf).convert("RGB")

        input_mask = mask_transform(input_mask)
        detail = img_transform(detail.convert("RGB"))
        reference = img_transform(reference)
        
        input_mask = torch.unsqueeze(input_mask, 0)
        detail = torch.unsqueeze(detail, 0)
        reference = torch.unsqueeze(reference,0)
        
        with torch.no_grad():
        
          st_time=time.time()
          model.set_input(detail,input_mask,reference)
          model.forward()
          ed_time=time.time()
          cost_time=ed_time-st_time
          
          fake_out = model.out
          fake_out = fake_out.detach().cpu() * input_mask + detail*(1-input_mask)
          fake_image = (fake_out+1)/2.0
          
        all_time +=cost_time
        count +=1

        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(results_dir+filename+"_result.jpg")
    
        input, reference, output, GT = model.get_current_visuals()
        image_out = torch.cat([input,reference,output,GT], 0)
        grid = torchvision.utils.make_grid(image_out)
        writer.add_image('picture(%d)' % i,grid,i)
        
    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time/(count)))
