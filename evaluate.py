# reference: https://github.com/nie-lang/UDIS2/blob/main/Warp/Codes/test.py
import sys
# from attr import validate
import argparse
import numpy as np
import torch
import torch.nn as nn
import importlib

import skimage
import argparse
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from yacs.config import CfgNode as CN
sys.path.append('core')
import core.datasets as datasets
from core.flowHomoAdpater import FlowHomoAdpater
from core.FlowFormer import build_flowformer
from core.UDIS2.Homography.network import UDIS2Network


@torch.no_grad()
def validate_with_model(cfg, val_dataset, model, num_size=12):
    """ Evaluate """

    total_steps = 0
    print(" model.training", model.training)
    print("len val_dataset",len(val_dataset))
    
    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    file_name_list = []
    data_loader = DataLoader(val_dataset, batch_size=num_size, shuffle=False, num_workers=num_size, drop_last=False)

    for idx, (image1, image2, info) in enumerate(data_loader):
        torch.cuda.empty_cache()

        image1 = image1.cuda()
        image2 = image2.cuda()

        output = {}
        out_dict = model(image1, image2, type="test_eval")
        warped_image_pred = out_dict["final_warp_output"][:,0:3,...]
        valid = out_dict["final_warp_output"][:,3:6,...].mean(dim=1, keepdim=True)


        """ calculate psnr/ssim """
        image1_cpu = image1.detach().cpu()
        warp_mesh_mask_cpu = valid.detach().cpu()
        warp_mesh_cpu = warped_image_pred.detach().cpu()
        
        def calculate_metrics(sample_idx):
            inpu1_np = image1_cpu[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8).numpy()
            warp_mesh_mask_np = warp_mesh_mask_cpu[sample_idx].detach().cpu().repeat(3,1,1).permute(1,2,0).to(torch.uint8).numpy() #H,W
            warp_mesh_np = warp_mesh_cpu[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8).numpy()
            ## calculate psnr/ssim
            psnr = skimage.metrics.peak_signal_noise_ratio(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255)
            ssim = skimage.metrics.structural_similarity(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255, multichannel=True)
            filename = info[0][sample_idx].split("/")[-1]
            ith = total_steps + sample_idx
            return psnr, ssim, filename, ith
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(calculate_metrics, range(image1.shape[0]))
        for i, (psnr, ssim, filename, ith) in enumerate(results):
            print('i = {}, psnr = {:.6f}'.format(ith, psnr))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            file_name_list.append(filename)
            total_steps += 1
        
    print("=================== Analysis ==================")
    print("Number of Test", len(psnr_list))
    print("[psnr]")
    psnr_list.sort(reverse = True)
    psnr_list_30 = psnr_list[0 : 331]
    psnr_list_60 = psnr_list[331: 663]
    psnr_list_100 = psnr_list[663: -1]
    print("top 30%: {:.6f}".format( np.mean(psnr_list_30)))
    print("top 30~60%: {:.6f}".format( np.mean(psnr_list_60)))
    print("top 60~100%: {:.6f}".format( np.mean(psnr_list_100)))
    print('average psnr: {:.6f}'.format( np.mean(psnr_list)))

    print("\n[ssim]")
    ssim_list.sort(reverse = True)
    ssim_list_30 = ssim_list[0 : 331]
    ssim_list_60 = ssim_list[331: 663]
    ssim_list_100 = ssim_list[663: -1]
    print("top 30%: {:.6f}".format( np.mean(ssim_list_30)))
    print("top 30~60%: {:.6f}".format( np.mean(ssim_list_60)))
    print("top 60~100%: {:.6f}".format( np.mean(ssim_list_100)))
    print('average ssim: {:.6f}'.format( np.mean(ssim_list)))
    print("##################end testing#######################")

    # logger.close()
    result_dict = {
        "avg_psnr": np.mean(psnr_list),
        "avg_ssim": np.mean(ssim_list),
        "easy_psnr": np.mean(psnr_list_30),
        "mid_psnr": np.mean(psnr_list_60),
        "hard_psnr": np.mean(psnr_list_100),
        "easy_ssim": np.mean(ssim_list_30),
        "mid_ssim": np.mean(ssim_list_60),
        "hard_ssim": np.mean(ssim_list_100),
    }
    return result_dict



@torch.no_grad()
def validate(cfg, val_dataset):
    """ Model """
    assert cfg.restore_ckpt is not None, "Please specify the checkpoint using in restore_ckpt"
    if hasattr(cfg,"homo_backbone") and cfg.homo_backbone != None:
        homo_backbone = UDIS2Network(only_homo = True)
        flow_backbone = build_flowformer(cfg)
        print("FlowHomoAdpater")
        model = nn.DataParallel(FlowHomoAdpater(homo_backbone=homo_backbone ,flow_backbone = flow_backbone, cfg= cfg))
    else:
        raise NotImplementedError
    print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
    model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.eval()

    """ Evaluate """
    return validate_with_model(cfg=cfg, val_dataset=val_dataset ,model=model)

if __name__ == '__main__':
    """ Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/final_ckpt", help="ckpt path")
    parser.add_argument('--model_config_name', type=str, default="last_config", help="model config")
    parser.add_argument('--data_dir', type=str, default="./data/UDIS/UDIS-D/", help="data dir")
    args = parser.parse_args()
    
    """ CONFIG """
    init_config_dict = importlib.import_module(f"configs.{args.model_config_name}").config_dict
    ckpt_path = args.ckpt_path
    cfg = CN(init_dict=init_config_dict)
    cfg.batch_size = 1
    cfg.restore_ckpt = ckpt_path

    """ Dataset """
    data_dir = args.data_dir
    val_dataset = datasets.UDISDataset(data_dir=data_dir, aug_params=None, phase="testing")

    """ Evaluate """
    result_dict = validate(cfg, val_dataset)
    print(result_dict)
