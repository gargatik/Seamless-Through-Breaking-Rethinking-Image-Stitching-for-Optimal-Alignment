import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from core.inference.utils import get_border_point_on_valid_mask, get_point_pairs, shift_points, boundary_src_and_tgt, to_pillow_fn
from core.inference.vis_utils import plot_quiver
# different tps methods
# from core.inference.tps_methods.kornia_tps import get_tps_transform, warp_image_tps
from core.inference.tps_methods.opencv_tps import tensor2WarpImage_TPS
from core.inference.tps_methods.other_tps import tensor2_warp_image_cv
# different sample point methods
from core.inference.sample_point_methods import advanced_uniform_sample_border_points



# preprocess -> sample_point -> warp_by_tps -> mix_tps_warp(inpainting) -> output
@torch.no_grad()
def tps_H_warp(inputs, image_limit, tps_pipeline_config, inpaint_fn=None, is_plot=False):
    """
    Apply the Thin Plate Spline (TPS) transformation to warp the result already warped by homography, and then perform inpainting to blend it.

    Args:
        inputs (object): The input object containing various tensors.
        image_limit (object): The object containing image limit variables.
        tps_pipeline_config (object): The object containing TPS pipeline configuration variables.
        inpaint_fn (function, optional): The function used for inpainting. Defaults to None. if None, will use default inpainting function.
        is_plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        object: The output object containing the warped tensors.
    """

    # inputs
    output1 = inputs.output1
    mask1 = inputs.mask1
    H_warp = inputs.H_warp
    H_warp_mask = inputs.H_warp_mask
    final_warp = inputs.final_warp
    mask2 = inputs.mask2
    residual_flow = inputs.residual_flow
    valid = inputs.valid
    occlusion_mask = inputs.occlusion_mask
    border_points_mask = inputs.border_points_mask

    # image_limit 
    width_min = image_limit.width_min
    height_min = image_limit.height_min
    out_height = image_limit.out_height
    out_width = image_limit.out_width

    # tps_pipeline_config 
    grid_h = tps_pipeline_config.grid_h
    grid_w = tps_pipeline_config.grid_w
    pad_num = tps_pipeline_config.pad_num
    residual_flow_use_forward = tps_pipeline_config.residual_flow_use_forward
    flow_limit = tps_pipeline_config.flow_limit
    add_corner = tps_pipeline_config.add_corner
    get_pt_methods = tps_pipeline_config.get_pt_methods
    add_meshgrid = tps_pipeline_config.add_meshgrid
    affine_scale = tps_pipeline_config.affine_scale
    kernel_scale = tps_pipeline_config.kernel_scale
    use_boundary_limit = tps_pipeline_config.use_boundary_limit
    tps_method = tps_pipeline_config.tps_method
    output2_is_only_tps = tps_pipeline_config.output2_is_only_tps
    do_avg_pooling = tps_pipeline_config.do_avg_pooling

   

    """ Preprocess the residaul_flow and H_warp to get tps_H_warp"""
    preprocess_config = {
        "do_avg_pooling": do_avg_pooling,
        "residual_flow_use_forward": residual_flow_use_forward,
        "grid_h": grid_h,   
        "grid_w": grid_w
    }
    residual_flow = preprocess(residual_flow, valid, **preprocess_config, is_plot=is_plot)
    
    # Flow (origin resolution)
    W, H = residual_flow.shape[-1], residual_flow.shape[-2]
    width_max = out_width - abs(width_min)
    height_max = out_height - abs(height_min)
    flow_size = (residual_flow.shape[2], residual_flow.shape[3])

    """ Sample points """
    sample_points_config = {
        "out_height": out_height,
        "out_width": out_width,
        "width_min": width_min,
        "height_min": height_min,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "pad_num": pad_num,
        "get_pt_methods": get_pt_methods,
        "flow_limit": flow_limit,
        "H_warp": H_warp,
        "occlusion_mask": occlusion_mask,
        "valid": valid,
    }
    src_points, target_points, points_src, points_dst = sample_init_points(residual_flow, **sample_points_config, is_plot=is_plot)
    if use_boundary_limit:
        # filter points by boundary limit
        points_src, points_dst = boundary_src_and_tgt(points_src, points_dst, target_points, out_height=out_height, out_width=out_width)
    if add_corner:
        # add 4 corner pt
        points_src = torch.cat((points_src, torch.tensor([[[0,0],[0,out_height-1],[out_width-1,0],[out_width-1,out_height-1]]]).to(points_src.device).repeat(points_src.shape[0],1,1)), dim=1)
        points_dst = torch.cat((points_dst, torch.tensor([[[0,0],[0,out_height-1],[out_width-1,0],[out_width-1,out_height-1]]]).to(points_dst.device).repeat(points_dst.shape[0],1,1)), dim=1)
    if border_points_mask is not None:
        # filter points by border_points_mask
        # border_points_mask is the shape of (B, 1, H, W)
        # points_src and points_dst is the shape of (B, N, 2)
        # if border_points_mask  == 1, then keep it
        assert border_points_mask.shape[0] == 1
        assert border_points_mask.shape[1] == 1
        assert border_points_mask.shape[-2:] == (out_height, out_width)
        _border_points_mask = border_points_mask.squeeze(0).squeeze(0)
        # keep_idx
        keep_idx = []
        for i in range(src_points.shape[1]):
            sx, sy = points_src[0,i,:].long()
            if _border_points_mask[sy, sx] == 1:
                keep_idx.append(i)
        keep_idx = torch.tensor(keep_idx)
        points_src = points_src[:, keep_idx, :]
        points_dst = points_dst[:, keep_idx, :]
    
    """ TPS Warp """

    warp_by_tps_config = {
        "tps_method": tps_method,
        "kernel_scale": kernel_scale,
        "affine_scale": affine_scale,
        "out_height": out_height,
        "out_width": out_width,
    }
    tps_H_warp_and_tps_H_warp_mask  = warp_by_tps(H_warp, H_warp_mask, points_src, points_dst, **warp_by_tps_config, is_plot=is_plot)
    tps_H_warp, tps_H_warp_mask = tps_H_warp_and_tps_H_warp_mask[:,0:3,:,:], tps_H_warp_and_tps_H_warp_mask[:,3:,:,:]
    tps_H_warp_mask = tps_H_warp_mask.mean(dim=1, keepdim=True) 
    tps_H_warp_mask = (tps_H_warp_mask >= 0.5).float() 
    inv_tps_H_warp_mask = 1.0 - tps_H_warp_mask
    inv_tps_H_warp_mask_np = inv_tps_H_warp_mask[0,0].detach().cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    inv_tps_H_warp_mask_np = cv2.erode(inv_tps_H_warp_mask_np, kernel)     
    inv_tps_H_warp_mask_np = cv2.dilate(inv_tps_H_warp_mask_np, kernel)    
    inv_tps_H_warp_mask = torch.tensor(inv_tps_H_warp_mask_np).unsqueeze(0).unsqueeze(0)
    tps_H_warp_mask = 1.0 - inv_tps_H_warp_mask
    tps_H_warp *= tps_H_warp_mask


    # Mix tps_H_warp and final_warp
    final_warp_mask = (final_warp >= 3).float().mean(dim=1, keepdim=True)
    final_warp_mask = (final_warp_mask >= 0.5).float()
    
    invert_mask1 = (1 - mask1).float().mean(dim=1, keepdim=True)
    invert_mask1 = (invert_mask1 >= 0.5).float()
    tps_final_warp = final_warp * final_warp_mask + tps_H_warp * (1 - final_warp_mask) * invert_mask1
    tps_final_warp_mask = final_warp_mask + (1 - final_warp_mask) * tps_H_warp_mask * invert_mask1

    # blend image
    output1 = output1.detach().cpu().to(torch.float32)
    mask1 = mask1.detach().cpu().to(torch.float32)
    output2 = (tps_final_warp.detach() * tps_final_warp_mask).cpu().to(torch.float32)  
    mask2 = tps_final_warp_mask.detach().cpu().to(torch.float32)
    
    mix_tps_flow_warp = output2.clone().detach().cpu().to(torch.float32)
    mix_tps_flow_warp_mask = mask2.clone().detach().cpu().to(torch.float32)

    new_blend_image = ( output1 * mask1 + output2 * mask2 ) / (mask1 + mask2)
    new_blend_image = new_blend_image.clip(0,255).to(torch.uint8)

    if output2_is_only_tps:
        output2 = tps_H_warp.detach().cpu().to(torch.float32) * tps_H_warp_mask.detach().cpu().to(torch.float32)
        mask2 = tps_H_warp_mask.detach().cpu().to(torch.float32)

    """ INPAINT """
    if inpaint_fn is not None:
        assert output2_is_only_tps == True
        tps_H_warp = output2.clone()
        tps_H_warp_mask = mask2.clone()
        padding = (int(abs(width_min)),int( abs(width_max - W)), int(abs(height_min)), int(abs(height_max - H)))
        tps_final_warp, tps_final_warp_mask, inpaint_img, inpaint_img_mask, inpaint_area_mask = inpaint_fn(tps_H_warp = tps_H_warp, tps_H_warp_mask = tps_H_warp_mask, output1=output1, mask1=mask1, final_warp=final_warp, occlusion_mask=occlusion_mask, padding=padding, residual_flow=residual_flow)
        output2 = tps_final_warp.clone()
        mask2 = tps_final_warp_mask.clone()
        new_blend_image = ( output1 * mask1 + output2 * mask2 ) / (mask1 + mask2)
        new_blend_image = new_blend_image.clip(0,255).to(torch.uint8)
    
    """ Result """
    out_dict = {}

    out_dict.update({
        "new_blend_image": new_blend_image,
        "tps_output": tps_H_warp,
        "output2": output2,
        "mix_tps_flow_warp": mix_tps_flow_warp,
        "mix_tps_flow_warp_mask": mix_tps_flow_warp_mask,
        "mask2": mask2,
    })

    if inpaint_fn is not None:
        out_dict.update({
            "inpaint_img": inpaint_img,
            "inpaint_area_mask": inpaint_area_mask
        })
    return out_dict
    

    




def preprocess(residual_flow, valid, do_avg_pooling, residual_flow_use_forward, grid_h, grid_w, is_plot=False):
    """
    Preprocesses the residual flow tensor.

    Args:
        residual_flow (torch.Tensor): The residual flow tensor.
        valid (torch.Tensor or None): The validity mask tensor. If provided, the residual flow will be multiplied by the validity mask.
        do_avg_pooling (bool): Whether to perform average pooling on the residual flow.
        residual_flow_use_forward (bool): Whether to use the forward residual flow.
        grid_h (int): The height of the grid.
        grid_w (int): The width of the grid.
        is_plot (bool, optional): Whether to plot the residual flow. Defaults to False.

    Returns:
        torch.Tensor: The preprocessed residual flow tensor.
    """
    
    if do_avg_pooling:
        origin_flow = residual_flow.clone()
        kernel_size = min(grid_h,grid_w) // 2 * 2 - 1
        padding = (kernel_size - 1) // 2
        residual_flow = F.pad(residual_flow, (padding,padding,padding,padding), mode='constant')
        residual_flow = F.avg_pool2d(residual_flow, kernel_size=kernel_size, stride=1, padding=0)
        residual_flow = F.interpolate(residual_flow, size=origin_flow.shape[-2:], mode='bilinear', align_corners=False)

    if not residual_flow_use_forward:
        residual_flow = -residual_flow.clone()
        
    if valid is not None:
        residual_flow *= valid

    return residual_flow


def sample_init_points(residual_flow, out_height, out_width, width_min, height_min, grid_h, grid_w, pad_num, get_pt_methods, flow_limit, H_warp, occlusion_mask, valid, is_plot=False):
    """
    Sample initial points on the border for the TPS (Thin Plate Spline) pipeline.

    Args:
        residual_flow (torch.Tensor): The residual flow tensor.
        out_height (int): The output height.
        out_width (int): The output width.
        width_min (int): The minimum width.
        height_min (int): The minimum height.
        grid_h (int): The grid height.
        grid_w (int): The grid width.
        pad_num (int): The padding number.
        get_pt_methods (list): A list of methods to get the point pairs.
        flow_limit (float): The flow limit.
        H_warp (torch.Tensor): The warped image tensor.
        occlusion_mask (torch.Tensor): The occlusion mask tensor.
        valid (torch.Tensor): The valid mask tensor.
        is_plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        torch.Tensor: The source points tensor. (1, N, 2)
        torch.Tensor: The target points tensor. (1, N, 2)
        torch.Tensor: The source points (shifted) tensor. (1, N, 2)
        torch.Tensor: The target points (shifted) tensor. (1, N, 2)
    """

    # Flow (origin resolution)
    W, H = residual_flow.shape[-1], residual_flow.shape[-2]
    width_max = out_width - abs(width_min)
    height_max = out_height - abs(height_min)
    flow_size = (residual_flow.shape[2], residual_flow.shape[3])

    # Sample few points on the border
    src_points, target_points = None, None
    non_shifted_src_points, non_shifted_target_points = None, None
    for get_pt_method in get_pt_methods:
        _src_points, _target_points = None, None
        if get_pt_method == "advanced_uniform":
            print("advanced_uniform")
            padding = (int(abs(width_min)),int( abs(width_max - W)), int(abs(height_min)), int(abs(height_max - H)))
            # crop H_warp by padding
            step = max(H,W) // min(grid_h, grid_w)
            cropped_H_warp = torchvision.transforms.functional.crop(H_warp, top=padding[2], left=padding[0], height=H, width=W)
            border_points = advanced_uniform_sample_border_points(cropped_H_warp, step=step, pad_num=pad_num)
            # Get source and target point pairs
            _src_points, _target_points = get_point_pairs(border_points, residual_flow, flow_limit)
        elif get_pt_method == "advanced_uniform_multi":
            print("advanced_uniform_multi")
            padding = (int(abs(width_min)),int( abs(width_max - W)), int(abs(height_min)), int(abs(height_max - H)))
            # crop H_warp by padding
            step = max(H,W) // min(grid_h, grid_w)
            cropped_H_warp = torchvision.transforms.functional.crop(H_warp, top=padding[2], left=padding[0], height=H, width=W)
            border_points = advanced_uniform_sample_border_points(cropped_H_warp, step=step, pad_num=pad_num)
            pad_num_list = []
            p_num = step
            while(p_num <= (max(H,W)//4)):
                pad_num_list.append(p_num)
                p_num= p_num * 2
            # pad_num_list = [step, step*2]
            for pd_num in pad_num_list:
                _border_points = advanced_uniform_sample_border_points(cropped_H_warp, step=step, pad_num=pd_num)
                border_points = torch.cat((border_points, _border_points), dim=0)
            # Get source and target point pairs
            _src_points, _target_points = get_point_pairs(border_points, residual_flow, flow_limit)
        else:
            raise NotImplementedError
        if src_points is None:
            src_points = _src_points
            target_points = _target_points
        else:
            src_points = torch.cat((src_points, _src_points), dim=1)
            target_points = torch.cat((target_points, _target_points), dim=1)
         
        # Shift the point pairs to match new resolution
        if src_points is None and non_shifted_src_points is None:
            raise Exception("src_points is None and non_shifted_src_points is None")
        
    if src_points is not None:
        carried_shfit_fn = lambda x : shift_points(x, width_min, width_max, height_min, height_max, H, W, pad_num)
        points_src = carried_shfit_fn(src_points)
        points_dst = carried_shfit_fn(target_points)
        if non_shifted_src_points is not None:
            points_src = torch.cat((points_src, non_shifted_src_points), dim=1)
            points_dst = torch.cat((points_dst, non_shifted_target_points), dim=1)    
    else:
        points_src = non_shifted_src_points
        points_dst = non_shifted_target_points
   
    return src_points, target_points, points_src, points_dst


def warp_by_tps(H_warp, H_warp_mask, points_src, points_dst, out_height, out_width, tps_method, kernel_scale, affine_scale, is_plot=False):
    """
    Apply Thin Plate Splines (TPS) warp on the input image using the given parameters.

    Args:
        H_warp (torch.Tensor): The input image to be warped.
        H_warp_mask (torch.Tensor): The mask of the input image.
        points_src (torch.Tensor): The source points for the TPS warp.
        points_dst (torch.Tensor): The destination points for the TPS warp.
        out_height (int): The height of the output image.
        out_width (int): The width of the output image.
        tps_method (str): The method to use for TPS warp. Options: "kornia", "opencv", "method3".
        kernel_scale (float): The scale factor for the kernel weights.
        affine_scale (float): The scale factor for the affine weights.
        is_plot (bool, optional): Whether to plot debug information. Defaults to False.

    Returns:
        torch.Tensor: The TPS warped image.

    Raises:
        NotImplementedError: If the specified TPS method is not implemented.
    """
    
    H_warp_and_H_warp_mask = torch.cat((H_warp, H_warp_mask), dim=1)
    if tps_method == "kornia":
        from core.inference.tps_methods.kornia_tps import get_tps_transform, warp_image_tps

        points_src = points_src.to(torch.float64)
        points_dst = points_dst.to(torch.float64)
        points_src[:,:,0] = points_src[:,:,0] / out_width
        points_src[:,:,1] = points_src[:,:,1] / out_height
        points_dst[:,:,0] = points_dst[:,:,0] / out_width
        points_dst[:,:,1] = points_dst[:,:,1] / out_height
        
        points_src = points_src.to(torch.float32)
        points_dst = points_dst.to(torch.float32)

        # TPS warp on H_warp to get tps_H_warp
        # # note that we are getting the reverse transform: dst -> src
        kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
        kernel_weights = kernel_weights * kernel_scale
        affine_weights = affine_weights * affine_scale
        tps_H_warp_and_tps_H_warp_mask = warp_image_tps(H_warp_and_H_warp_mask, points_src, kernel_weights, affine_weights, align_corners=False)
       
    elif tps_method == "opencv":
        # opencv
        tps_H_warp_mask, new_pts1, new_pts2  = tensor2WarpImage_TPS(points_src,points_dst,H_warp_mask) 
        tps_H_warp, new_pts1, new_pts2  = tensor2WarpImage_TPS(points_src,points_dst,H_warp) 
        tps_H_warp_and_tps_H_warp_mask = torch.cat((tps_H_warp, tps_H_warp_mask), dim=1)
        tps_H_warp_and_tps_H_warp_mask = tps_H_warp_and_tps_H_warp_mask.float()
        # warped_points_src = new_pts1
        warped_points_src = torch.zeros_like(points_dst)

        points_src = points_src.to(torch.float64)
        points_dst = points_dst.to(torch.float64)
        warped_points_src = warped_points_src.to(torch.float64)
        points_src[:,:,0] = points_src[:,:,0] / out_width
        points_src[:,:,1] = points_src[:,:,1] / out_height
        points_dst[:,:,0] = points_dst[:,:,0] / out_width
        points_dst[:,:,1] = points_dst[:,:,1] / out_height
        warped_points_src[:,:,0] = warped_points_src[:,:,0] / out_width
        warped_points_src[:,:,1] = warped_points_src[:,:,1] / out_height
        points_src = points_src.to(torch.float32)
        points_dst = points_dst.to(torch.float32)
        warped_points_src = warped_points_src.to(torch.float32)

    elif tps_method == "other":
        ## other
        points_src = points_src.to(torch.float64)
        points_dst = points_dst.to(torch.float64)
        # normalize
        points_src[:,:,0] = points_src[:,:,0] / out_width
        points_src[:,:,1] = points_src[:,:,1] / out_height
        points_dst[:,:,0] = points_dst[:,:,0] / out_width
        points_dst[:,:,1] = points_dst[:,:,1] / out_height

        points_src = points_src.to(torch.float32)
        points_dst = points_dst.to(torch.float32)
        # TPS warp on H_warp to get tps_H_warp
        tps_H_warp_mask = tensor2_warp_image_cv(H_warp_mask, points_src, points_dst)
        tps_H_warp = tensor2_warp_image_cv(H_warp, points_src, points_dst)
        tps_H_warp_and_tps_H_warp_mask = torch.cat((tps_H_warp, tps_H_warp_mask), dim=1)
        tps_H_warp_and_tps_H_warp_mask = tps_H_warp_and_tps_H_warp_mask.float()
        warped_points_src = torch.zeros_like(points_dst)
    else: 
        raise NotImplementedError
    
    return tps_H_warp_and_tps_H_warp_mask


