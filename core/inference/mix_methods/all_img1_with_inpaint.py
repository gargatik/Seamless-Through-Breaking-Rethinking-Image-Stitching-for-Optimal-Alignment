import torch
import matplotlib.pyplot as plt
from core.inference.utils import to_pillow_fn, dilate_thin_area, dilate_mask
import importlib


# all_img1_with_inpaint
def mix_fn(tps_H_warp, tps_H_warp_mask,  output1, mask1, final_warp, occlusion_mask, padding, residual_flow,  use_composition=False, is_plot=False, resize_to_area_limit_before_inpaint=950*950, inpainter=None):
    """
    Perform image mixing and inpainting.
    with image1 as the main source of inpainting.

    Args:
        tps_H_warp (torch.Tensor): Tensor representing the warped image.
        tps_H_warp_mask (torch.Tensor): Tensor representing the mask of the warped image.
        output1 (torch.Tensor): Tensor representing the output image.
        mask1 (torch.Tensor): Tensor representing the mask of the output image.
        final_warp (torch.Tensor): Tensor representing the final warped image.
        occlusion_mask (torch.Tensor): Tensor representing the occlusion mask.
        padding (int): Padding value.
        residual_flow (bool): Flag indicating whether to use residual flow.
        use_composition (bool, optional): Flag indicating whether to use composition. Defaults to False.
        is_plot (bool, optional): Flag indicating whether to plot intermediate results. Defaults to False.
        resize_to_area_limit_before_inpaint (int, optional): Area limit for resizing before inpainting. Defaults to 950*950.
        inpainter (module, optional): Inpainter module. Defaults to None.

    Returns:
        torch.Tensor: Tensor representing the final warped image.
        torch.Tensor: Tensor representing the mask of the final warped image.
        torch.Tensor: Tensor representing the inpainted image.
        torch.Tensor: Tensor representing the mask of the inpainted image.
        torch.Tensor: Tensor representing the mask of the inpainted area.
    """

    if use_composition:
        print("[Warning]: use_composition is not implemented")

    if inpainter is None:
        inpainter = importlib.import_module(f"core.inference.mix_methods.utils.inpainter").inpainter

    device = tps_H_warp.device
    to_device_fn = lambda x : x.to(device)
    inv_mask1 = 1. - torch.where(mask1 > 0.5 , torch.ones_like(mask1),  torch.zeros_like(mask1) )
    
    tps_final_warp = final_warp * occlusion_mask * mask1 + tps_H_warp  * inv_mask1
    tps_final_warp_mask = occlusion_mask * mask1 +  tps_H_warp_mask  * inv_mask1
    tps_final_warp, tps_final_warp_mask = list( map (to_device_fn, [tps_final_warp, tps_final_warp_mask]))

    # inpaint tps_final_warp
    ## inpaint_img_by_only_img1
    inpaint_area_mask = (1. - tps_final_warp_mask) * mask1 
    inpaint_area_mask = dilate_thin_area(inpaint_area_mask)

    inpaint_area_mask_dilated = dilate_mask(inpaint_area_mask, kernel_size=7)
    inpaint_area_mask_dilated = torch.where(inpaint_area_mask_dilated > 0 , torch.ones_like(inpaint_area_mask_dilated), torch.zeros_like(inpaint_area_mask_dilated))
    diff = inpaint_area_mask - inpaint_area_mask_dilated
    mask1_border = diff = torch.abs(diff)
    inpaint_area_mask = inpaint_area_mask_dilated
        
    inpaint_by_img1_mask = (1 - mask1_border) * inpaint_area_mask * mask1
    overlap_area_inpaint_by_img1 = output1 * inpaint_by_img1_mask

    overlap_area_inpaint = overlap_area_inpaint_by_img1 
    inpaint_img = tps_final_warp * (1-inpaint_by_img1_mask)  +  overlap_area_inpaint * inpaint_by_img1_mask
    inpaint_img_mask = tps_final_warp_mask * (1-inpaint_by_img1_mask) + mask1 * inpaint_by_img1_mask
    inpaint_img_mask = inpaint_img_mask.to(dtype=mask1.dtype)
    inpaint_img_mask = torch.where( inpaint_img_mask >0.5, torch.ones_like(inpaint_img_mask), torch.zeros_like(inpaint_img_mask))

    inpaint_img_by_only_img1 = inpaint_img.clone().detach()

    ## inpaint img by inpainting other area
    output2_mask = tps_H_warp_mask * inv_mask1 + mask1
    output2_mask = torch.where(output2_mask > 0.5 , torch.ones_like(output2_mask), torch.zeros_like(output2_mask))
    inpaint_by_other_mask = (1. - inpaint_by_img1_mask) * mask1_border #* output2_mask
    inpaint_by_other_mask = dilate_thin_area(inpaint_by_other_mask, thickening_kernel_size=8)

    img_mask_threshold = 0.05
    inpaint_by_other_mask = torch.where( inpaint_by_other_mask > img_mask_threshold, torch.ones_like(inpaint_by_other_mask), torch.zeros_like(inpaint_by_other_mask))
    inpaint_img = inpaint_img * (1- inpaint_by_other_mask)
    
    ### inpainting
    if (inpaint_by_other_mask.shape[2] * inpaint_by_other_mask.shape[3] > resize_to_area_limit_before_inpaint )or inpainter.name == "gan_inpainter":
        print("[Warning]: resize for inpaint!!")
        if(inpainter.name== "transref_inpainter"):
            control_image_tensor = inpaint_img_by_only_img1.clip(0,255).to(dtype=inpaint_img.dtype, device=inpaint_img.device)
            inpaint_img = control_image_tensor
            inpaint_img = inpainter.inpaint(inpaint_img, inpaint_by_other_mask, control_image_tensor=control_image_tensor, resize_to_area_limit_before_inpaint=False)
        else:
            inpaint_img = inpainter.inpaint(inpaint_img, inpaint_by_other_mask, resize_to_area_limit_before_inpaint=resize_to_area_limit_before_inpaint)
    else:

        if(inpainter.name== "transref_inpainter"):
            control_image_tensor = inpaint_img_by_only_img1.clip(0,255).to(dtype=inpaint_img.dtype, device=inpaint_img.device)
            inpaint_img = control_image_tensor
            other_inpaint_img = inpainter.inpaint(inpaint_img, inpaint_by_other_mask, control_image_tensor=control_image_tensor, resize_to_area_limit_before_inpaint=False)
        else:
            other_inpaint_img = inpainter.inpaint(inpaint_img, inpaint_by_other_mask, resize_to_area_limit_before_inpaint=False)

        inpaint_inpaint_img = other_inpaint_img 
        inpaint_img = inpaint_inpaint_img.to(dtype=inpaint_img.dtype, device=inpaint_img.device)
    overlap_area_inpaint_by_other = inpaint_img * inpaint_by_other_mask 
    inpaint_img= inpaint_img.detach().cpu().float().to(mask1.device)
    inpaint_img_mask = tps_H_warp_mask
    inpaint_img = inpaint_img * inpaint_img_mask
    
    # result
    if(torch.count_nonzero(inpaint_img)==0):
        print("Warning: inpaint_img is all zero, not use!!")
    else:
        tps_final_warp = inpaint_img.clone()
        tps_final_warp_mask = inpaint_img_mask.clone()
    inpaint_area_mask =  torch.cat( (inpaint_img_by_only_img1, inpaint_by_other_mask[:,0:1,:,:] ), dim=1) # shape (B, 4, H, W)
    tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask = list( map (to_device_fn, [tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask]))
    return tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask