import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.inference.utils import to_pillow_fn, dilate_thin_area
import importlib

# inpaint_all_area
def mix_fn(tps_H_warp, tps_H_warp_mask, output1, mask1, final_warp, occlusion_mask, padding, residual_flow, use_composition=False, is_plot=False, resize_to_area_limit_before_inpaint=950*950, inpainter=None):
    """
    Perform mixing of images using various methods.
    with inpainting all area.
    
    Args:
        tps_H_warp (torch.Tensor): The TPS warped image.
        tps_H_warp_mask (torch.Tensor): The mask of the TPS warped image.
        output1 (torch.Tensor): The output image.
        mask1 (torch.Tensor): The mask of the output image.
        final_warp (torch.Tensor): The final warped image.
        occlusion_mask (torch.Tensor): The occlusion mask.
        padding (int): The padding value.
        residual_flow (torch.Tensor): The residual flow.
        use_composition (bool, optional): Whether to use composition. Defaults to False.
        is_plot (bool, optional): Whether to plot intermediate results. Defaults to False.
        resize_to_area_limit_before_inpaint (int, optional): The area limit for resizing before inpainting. Defaults to 950*950.
        inpainter (module, optional): The inpainter module. Defaults to None.

    Returns:
        torch.Tensor: The final warped image.
        torch.Tensor: The mask of the final warped image.
        torch.Tensor: The inpainted image.
        torch.Tensor: The mask of the inpainted image.
        torch.Tensor: The mask of the inpaint area.

    """
    if use_composition:
        print("[Warning]: use_composition is not implemented")

    print("inpaint_all_area")
    if inpainter is None:
        inpainter = importlib.import_module(f"core.inference.mix_methods.utils.inpainter").inpainter
    device = tps_H_warp.device
    to_device_fn = lambda x : x.to(device)
    inv_mask1 = 1. - mask1
    
    tps_final_warp = final_warp * occlusion_mask + tps_H_warp  * inv_mask1
    tps_final_warp_mask = occlusion_mask +  tps_H_warp_mask  * inv_mask1
    tps_final_warp, tps_final_warp_mask = list( map (to_device_fn, [tps_final_warp, tps_final_warp_mask]))


    # inpaint tps_final_warp
    inpaint_area_mask = (1. - tps_final_warp_mask) * mask1 * tps_H_warp_mask
    inpaint_area_mask = dilate_thin_area(inpaint_area_mask, thickening_kernel_size=16)
    ## inpainting
    if (inpaint_area_mask.shape[2] * inpaint_area_mask.shape[3] > resize_to_area_limit_before_inpaint ) or inpainter.name in ["gan_inpainter","transref_inpainter"]:
        print("Warning: resize for inpaint!!")
        if(inpainter.name== "transref_inpainter"):
            control_image_tensor = output1.clip(0,255).to(dtype=tps_final_warp.dtype, device=tps_final_warp.device)
            inpaint_img = tps_final_warp
            inpaint_img = inpainter.inpaint(inpaint_img, inpaint_area_mask, control_image_tensor=control_image_tensor, resize_to_area_limit_before_inpaint=False)
        else:
            inpaint_img = inpainter.inpaint(tps_final_warp, inpaint_area_mask, resize_to_area_limit_before_inpaint=resize_to_area_limit_before_inpaint)
    else:
        inpaint_img = inpainter.inpaint(tps_final_warp, inpaint_area_mask, resize_to_area_limit_before_inpaint=False)
    inpaint_img_mask = tps_H_warp_mask.clone()

    # result    
    if(torch.count_nonzero(inpaint_img)==0):
        print("Warning: inpaint_img is all zero, not use!!")
    else:
        tps_final_warp = inpaint_img.clone()
        tps_final_warp_mask = inpaint_img_mask.clone()
        
    tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask = list( map (to_device_fn, [tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask]))
    return tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask