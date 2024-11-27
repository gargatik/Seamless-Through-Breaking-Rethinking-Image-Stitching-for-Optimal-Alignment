import sys
sys.path.append('core')
import torch
import numpy as np
import importlib

torch.manual_seed(1234)
np.random.seed(1234)

""" CONFIG"""
import wandb
import argparse
from yacs.config import CfgNode as CN
# ARGS FOR SELECTING MODEL
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/final_ckpt", help="ckpt path")
    parser.add_argument('--model_config_name', type=str, default="last_config", help="model config")
    parser.add_argument("--data_root_path", type=str, default="./demo/")
    parser.add_argument("--txt_file", type=str, default="demo.txt")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--inf_cfg", type=str, default="all_img1_with_inpaint_g12_transRef")
    parser.add_argument("--result_dir", type=str, default="results")

    parser.add_argument("--skip_if_avg_fusion_exists", action="store_true", default=False, help="skip if avg_fusion exists")
    
    return parser.parse_args()

# WARPMODEL CONFIG
def get_warp_model_config(args):
    init_config_dict = importlib.import_module(f"configs.{args.model_config_name}").config_dict
    ckpt_path = args.ckpt_path
    cfg = CN(init_dict=init_config_dict)

    cfg.update(dict(args))
    cfg.batch_size = 1
    cfg.restore_ckpt = ckpt_path
    # cfg.suffix += "_{}".format(args.run.split("/")[-1])
    return cfg

# merge get_inference_config(), get_args() and get_warp_model_config(args)
def get_config():
    args = get_args() # parse by argparse, Namespace
    get_infernce_config = importlib.import_module(f"inf_configs.{args.inf_cfg}").get_infernce_config
    get_tps_pipline_config = importlib.import_module(f"inf_configs.{args.inf_cfg}").get_tps_pipline_config
    INFERENCE_CONFIG = get_infernce_config()
    cfg = CN(init_dict=INFERENCE_CONFIG)
    cfg.update(vars(args))
    final_cfg = get_warp_model_config(cfg)
    TPS_PIPELINE_CONFIG = get_tps_pipline_config(final_cfg)
    final_cfg.TPS_PIPELINE_CONFIG = TPS_PIPELINE_CONFIG

    return final_cfg
    
""" LOAD WARP MODEL """
import os
import torch.nn as nn
import wandb
from core.flowHomoAdpater import FlowHomoAdpater
from core.UDIS2.Homography.network import UDIS2Network
from core.FlowFormer import build_flowformer
def load_warp_model(cfg):
    only_init_model = cfg.only_init_model
    assert cfg.restore_ckpt is not None, "Please specify the checkpoint using in restore_ckpt"
    if hasattr(cfg,"homo_backbone") and cfg.homo_backbone != None:
        homo_backbone = UDIS2Network(only_homo = True)
        flow_backbone = build_flowformer(cfg)
        if only_init_model:
            if hasattr(cfg,"init_homo_ckpt") and cfg.init_homo_ckpt is not None:
                print("[Loading homo_backbone ckpt from {}]".format(cfg.init_homo_ckpt))
                homo_backbone.load_state_dict({k.replace('module.',''):v for k, v in torch.load(cfg.init_homo_ckpt)["model"].items()}, strict=True)
            if hasattr(cfg,"init_flow_ckpt") and cfg.init_flow_ckpt is not None:
                print("[Loading flow_backbone ckpt from {}]".format(cfg.init_flow_ckpt))
                flow_backbone.load_state_dict({k.replace('module.',''):v for k, v in torch.load(cfg.init_flow_ckpt).items()}, strict=True)
        print("FlowHomoAdpater")
        # model = nn.DataParallel(FlowHomoAdpater(homo_backbone=homo_backbone ,flow_backbone = flow_backbone, cfg= cfg))
        model = FlowHomoAdpater(homo_backbone=homo_backbone ,flow_backbone = flow_backbone, cfg= cfg)
        print("Deploy model to DataParallel")
        model =  nn.DataParallel(model)
        if only_init_model:
            print("only_init_model")
        else:
            print("[WARP NETWORK][Loading ckpt from {}]".format(cfg.restore_ckpt))
            model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)
        model.cuda()
        model.eval()
    else:
        raise NotImplementedError
    
    return model

""" LOAD COMPOSE MODEL """

def load_com_model(composition_model_path):
    from core.UDIS2.Composition.network import build_model, Network
    composition_model = Network()
    checkpoint = torch.load(composition_model_path)
    print("[COMPOSE NETWORK][Loading ckpt from {}]".format(composition_model_path))
    composition_model.load_state_dict(checkpoint['model'])
    composition_model.cuda()
    composition_model.eval()
    return composition_model, build_model

""" DATA LOADING """
def get_data_dict_list(data_root_path, txt_file):
    # {
    #     "DATA_PATH": data_path,
    #     "IMG1": "input1.jpg",
    #     "IMG2": "input2.jpg"
    # }
    data_dict_list = [] 
    result_pair_dir_list = data_root_path + txt_file
    with open(result_pair_dir_list, "r") as f:
        lines = f.readlines() 
        for line in lines:
            print("line",line)
            data_dir = os.path.join(data_root_path, line.strip())
            img1 = "input1.jpg"
            img2 = "input2.jpg"
            data_dict_list.append({"DATA_PATH": data_dir, "IMG1": img1, "IMG2": img2})
    print("data_dict_list",data_dict_list)
    return data_dict_list

import torchvision.transforms as T
import cv2
resize_512 = T.Resize((512,512))
# get_single_data
def loadSingleData(data_path, img1_name, img2_name, resize_to_512=False):
    # load image1
    input1 = cv2.imread(data_path+img1_name).astype('uint8')
    input1 = cv2.cvtColor(input1, cv2.COLOR_BGR2RGB)
    input1 = np.array(input1).astype(np.uint8)[..., :3]
    # load image2
    input2 = cv2.imread(data_path+img2_name).astype('uint8')
    input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    input2 = np.array(input2).astype(np.uint8)[..., :3]
    # convert to tensor
    input1_tensor = torch.from_numpy(input1).permute(2, 0, 1).float().unsqueeze(0)
    input2_tensor = torch.from_numpy(input2).permute(2, 0, 1).float().unsqueeze(0)

    if resize_to_512:
        input1_tensor = resize_512(input1_tensor)
        input2_tensor = resize_512(input2_tensor)

    return (input1_tensor, input2_tensor)

""" INFERENCE """
import shutil
from core.inference.tps_pipline import tps_H_warp
from core.inference.utils import to_pillow_fn
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from core.utils.other_utils import DictToObject

@torch.no_grad()
def inference_one_data(cfg, data_dict, save_root_path, error_list, warp_model, composition_model, compose_fn, inpainter):
    path  = data_dict["DATA_PATH"]
    img1_name = data_dict["IMG1"]
    img2_name =  data_dict["IMG2"]
    print("path",path, img1_name, img2_name)

    result_path = path.replace(cfg.data_root_path, save_root_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print("Create Result Dir", result_path)

    # copy image1 and image2
    src_input_path = path+img1_name
    dst_input_path = result_path+"input1."+img1_name.split('.')[-1]
    print(f"copy {src_input_path} to {dst_input_path}")
    shutil.copyfile(src_input_path, dst_input_path)

    src_input_path = path+img2_name
    dst_input_path = result_path+"input2."+img2_name.split('.')[-1]
    print(f"copy {src_input_path} to {dst_input_path}")
    shutil.copyfile(src_input_path, dst_input_path)

    """ START """
    torch.cuda.empty_cache()
    if True:
        image1, image2 = loadSingleData(data_path=path, img1_name = img1_name, img2_name= img2_name, resize_to_512=cfg.resize_to_512)
    
    if hasattr(cfg, "swap_image") and cfg.swap_image:
        # swap image1 and image2
        tmp = image1.clone()
        image1 = image2.clone()
        image2 = tmp
    info = [img1_name, img2_name]

    output = {}
    if True:
        # block raising an exception
        warp_model = warp_model.cuda()
        out_dict = warp_model(image1, image2, type="test_out", pad_mode=cfg.pad_mode)
        warp_model = warp_model.cpu()
    
    H_warp, final_warp, output1, output2, mask1, mask2, blend_image = out_dict["H_warp"], out_dict["final_warp"], out_dict["output1"], out_dict["output2"], out_dict["mask1"], out_dict["mask2"], out_dict["blend_image"]
    residual_flow = out_dict["residual_flow"]
    width_min, height_min, out_height, out_width = out_dict["width_min"], out_dict["height_min"], out_dict["out_height"], out_dict["out_width"]

    warp_input2_mask = out_dict["warp_input2_mask"].detach().cpu()
    occlusion_mask = out_dict["occlusion_mask"].detach().cpu()
    H_warp_mask = out_dict["H_warp_mask"].detach().cpu()

    # map all to deatch cpu
    H_warp = H_warp.detach().cpu()
    final_warp = final_warp.detach().cpu()
    output1 = output1.detach().cpu()
    output2 = output2.detach().cpu()
    mask1 = mask1.detach().cpu()
    mask2 = mask2.detach().cpu()
    blend_image = blend_image.detach().cpu()
    residual_flow = residual_flow.detach().cpu()
    valid = None
    if hasattr(cfg, "use_fb_consistency_mask") and cfg.use_fb_consistency_mask and cfg.TPS_PIPELINE_CONFIG.use_valid_on_flow:
        valid = out_dict["origin_occlusion_mask"].detach().cpu()
        # valid [0,1] -> soft half valid [0.5, 1]
        valid = (1 - valid) * 0.0 + valid 
       
    border_points_mask = None
    if hasattr(cfg, "use_fb_consistency_mask") and cfg.use_fb_consistency_mask and cfg.TPS_PIPELINE_CONFIG.use_border_points_mask:
        occlusion_mask = out_dict["occlusion_mask"].detach().cpu()
        if cfg.TPS_PIPELINE_CONFIG.use_occ_filter :
            border_points_mask = occlusion_mask
        else:
            border_points_mask = H_warp_mask.mean(dim=1,keepdim=True)
            border_points_mask = torch.where(border_points_mask > 0.5, torch.ones_like(border_points_mask), torch.zeros_like(border_points_mask))

    torch.cuda.empty_cache()
    # print("info",info)
    
    mix_fn = importlib.import_module(f"core.inference.mix_methods.{cfg.TPS_PIPELINE_CONFIG.mix_method}").mix_fn
    inpaint_fn = lambda **kwarg: mix_fn(**kwarg, inpainter=inpainter ,use_composition=cfg.TPS_PIPELINE_CONFIG.use_composition_when_inpaint, is_plot=cfg.TPS_PIPELINE_CONFIG.is_plot, resize_to_area_limit_before_inpaint= cfg.TPS_PIPELINE_CONFIG.resize_to_area_limit_before_inpaint)
     
    
    inputs = {
            "output1": output1,
            "mask1": mask1,
            "H_warp": H_warp,
            "H_warp_mask": H_warp_mask,
            "final_warp": final_warp,
            "mask2": mask2,
            "residual_flow": residual_flow,
            "valid": valid,
            "occlusion_mask": occlusion_mask,
            "border_points_mask": border_points_mask
        }
    inputs = DictToObject(inputs)
    image_limit = {
        "width_min": width_min,
        "height_min": height_min,
        "out_height": out_height,
        "out_width": out_width
    }
    image_limit = DictToObject(image_limit)

    tps_pipeline_config = cfg.TPS_PIPELINE_CONFIG
    new_out_dict = tps_H_warp(inputs, image_limit, tps_pipeline_config, inpaint_fn=inpaint_fn, is_plot=cfg.TPS_PIPELINE_CONFIG.is_plot)

    new_blend_image, tps_output, output2, mask2 = new_out_dict["new_blend_image"], new_out_dict["tps_output"], new_out_dict["output2"], new_out_dict["mask2"]
    
    # save image
    to_pillow_fn(H_warp).save(result_path + "H_warp.jpg")
    to_pillow_fn(final_warp).save(result_path + "flow_warp.jpg")
    to_pillow_fn(output1).save(result_path + "warp1.jpg")
    to_pillow_fn(output2).save(result_path + "warp2.jpg")
    
    mask1 = (mask1 > 0.5).float()
    Image.fromarray(mask1[0,0].detach().cpu().to(torch.uint8).numpy() * 255).save(result_path+"mask1.jpg")
    mask2 = (mask2 > 0.5).float()
    Image.fromarray(mask2[0,0].detach().cpu().to(torch.uint8).numpy() * 255).save(result_path+"mask2.jpg")
    to_pillow_fn(new_blend_image).save(result_path+"ave_fusion.jpg")
    
    if cfg.use_composition:
        def resize_process_fn(x):
            # resize tensors to large than 512
            if min(x.shape[2],x.shape[3]) < 512:
                scale_factor = 512 / min(x.shape[2],x.shape[3])
                x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            return x
        normalize_fn = lambda x : (x.clip(0, 255) / 127.5) - 1.0
        to_gpu_fn = lambda x : x.cuda()
        preprocess_fn = lambda x : normalize_fn(  resize_process_fn( to_gpu_fn(x) )  )
        preprocess_mask_fn = lambda x : resize_process_fn(to_gpu_fn(x))
        
        with torch.no_grad():
                composition_model = composition_model.cuda()
                batch_out = compose_fn(composition_model, preprocess_fn(output1), preprocess_fn(output2), preprocess_mask_fn(mask1), preprocess_mask_fn(mask2))
                composition_model = composition_model.cpu()
         

        stitched_image = batch_out['stitched_image']
        learned_mask1 = batch_out['learned_mask1']
        learned_mask2 = batch_out['learned_mask2']
        stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0).clip(0,255).astype(np.uint8)
        learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
        learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)
        
    if cfg.use_composition:
        # Image.fromarray(stitched_image).save( path_final_composition + filename)
        Image.fromarray(stitched_image).save( result_path + "composition.jpg")
        # cv2.imwrite(path_learn_mask1 + filename, learned_mask1)
        cv2.imwrite(result_path + "learned_mask1.jpg", learned_mask1)
        # cv2.imwrite(path_learn_mask2 + filename, learned_mask2)
        cv2.imwrite(result_path + "learned_mask2.jpg", learned_mask2)

    

    torch.cuda.empty_cache()


if __name__ == "__main__":
    """ PREPARE """
    cfg = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    
    warp_model = load_warp_model(cfg)

    if cfg.use_composition:
        composition_model, compose_fn = load_com_model(cfg.composition_model_path)
    
    data_dict_list = get_data_dict_list(cfg.data_root_path, cfg.txt_file)

    model_name= cfg.restore_ckpt.split("/")[-2] 
    resize_512_tag = "512" if cfg.resize_to_512 else ""
    surfix_name = f"{resize_512_tag}_{model_name}_{cfg.TPS_PIPELINE_CONFIG.get_pt_methods[0]}_{cfg.TPS_PIPELINE_CONFIG.mix_method}_g{cfg.TPS_PIPELINE_CONFIG.grid_h}"
    
    DST_ROOT_PATH = os.path.abspath(os.path.join(cfg.data_root_path ,f"../{cfg.result_dir}/"))
    save_root_path = f"{DST_ROOT_PATH}/ours_{surfix_name}/"

    # save experiment config to save_root_path
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    with open(save_root_path + "config.txt", "w") as f:
        f.write(cfg.dump()) 
    
    """ INPAINTER SETTING """
    inpainter = None
    if hasattr(cfg.TPS_PIPELINE_CONFIG, "inpainter") and cfg.TPS_PIPELINE_CONFIG.inpainter:
        print("use inpainter:", cfg.TPS_PIPELINE_CONFIG.inpainter)
        inpainter = importlib.import_module(f"core.inference.mix_methods.utils.{cfg.TPS_PIPELINE_CONFIG.inpainter}").inpainter
    else:
        print("use default inpainter")
        inpainter  = importlib.import_module(f"core.inference.mix_methods.utils.inpainter").inpainter # default inpainter
    
    """ INFERENCE """
    error_list = []
    print("start to inference")
    for data_dict in data_dict_list:
        if cfg.skip_if_avg_fusion_exists:
            if os.path.exists(save_root_path):
                print("[WARNING] Skip, Due to exist", save_root_path)
                continue
        inference_one_data(cfg, data_dict, save_root_path, error_list, warp_model, composition_model, compose_fn, inpainter)
    
    
    print("error_list", error_list)
    

