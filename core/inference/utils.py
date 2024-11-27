import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


to_pillow_fn = lambda x: Image.fromarray(x[0].permute(1,2,0).detach().cpu().to(torch.uint8).numpy())

def get_border_point_on_valid_mask(valid, grid_h, grid_w, pad_num=None, is_plot=False):
    # valid shape: (B, 1, H, W)
    # sobel filter the valid mask
    _, _, vh, vw = valid.shape
    valid = valid.float()
    valid = F.pad(valid, (1,1,1,1), mode='replicate')
    # horizontal sobel filter
    sobel_filter = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().to(valid.device)
    sobel_filter = sobel_filter.view(1,1,3,3).repeat(valid.shape[1],1,1,1)
    valid_horizontal = F.conv2d(valid, sobel_filter, padding=0, groups=valid.shape[1])
    # vertical sobel filter
    sobel_filter = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().to(valid.device)
    sobel_filter = sobel_filter.view(1,1,3,3).repeat(valid.shape[1],1,1,1)
    valid_vertical = F.conv2d(valid, sobel_filter, padding=0, groups=valid.shape[1])

    valid = valid_horizontal.abs() + valid_vertical.abs()

    valid = valid.abs() / valid.abs().max()
    # valid shape: (B, 1, H, W)
    # get the border point
    border_points = torch.nonzero(valid[0, 0, :, :] == 1)
    # border_points shape: (N, 2)
    # sample few points on the border


    # step = max(valid.shape[-1], valid.shape[-2]) // min(grid_h, grid_w)
    # border_points = border_points[::step]
    sample_num = grid_h * grid_w
    border_points_idx = np.random.choice(len(border_points), size=sample_num, replace=False)
    border_points = border_points[border_points_idx]


    # border_points (N, 2)--> (y ,x )
    
    if pad_num != None:
        # for every x,y , if x > W//2, x - pad_num, else x + pad_num,  also for y, H
        # x --> W
        border_points[:,1] = torch.where(border_points[:,1] > vw//2, border_points[:,1] - pad_num, border_points[:,1] + pad_num)
        # y --> H
        border_points[:,0] = torch.where(border_points[:,0] > vh//2, border_points[:,0] - pad_num, border_points[:,0] + pad_num)
    
        
    # y, x --> x, y
    border_points = border_points[:,[1,0]]

    return border_points



def get_point_pairs(border_points, flow, flow_limit):
    # border_points shape: (N, 2)
    # flow shape: (B, 2, H, W)
    # Get source and target point pairs
    B = flow.shape[0]
    src_points = border_points.unsqueeze(0).repeat(B, 1, 1) # (B, N, 2)
    # _flow = flow[:, :, border_points[:,0], border_points[:,1]].permute(0,2,1) # (B, 2, H, W) --> (B, N, 2) 
    _flow = flow[:, :, border_points[:,1], border_points[:,0]].permute(0,2,1) # (B, 2, H, W) --> (B, 2, N) --> (B, N, 2) 

    if flow_limit == -1: # flow_limit==-1 depend on resolution
        flow_limit = (flow.shape[2] + flow.shape[3]) // 2
        flow_limit = flow_limit // 8
        

    if flow_limit != None:
        # todo:
        # check flow is out of flow limit or not
        # if out of limit, not select this src_point
        # new method
        _flow_abs = _flow.abs()
        select_idx = (_flow_abs[:,:,0] < flow_limit) & (_flow_abs[:,:,1] < flow_limit) # (B, N)
        select_idx = select_idx.unsqueeze(-1).expand_as(_flow) # (B, N, 2)
        src_points = src_points[select_idx].view(B, -1, 2) # (B, N, 2)
        _flow = _flow[select_idx].view(B, -1, 2) # (B, N, 2)
        
        
        # old method
        # _flow = _flow.clip(-flow_limit, flow_limit)
    target_points = src_points + _flow 
    return src_points, target_points

def shift_points(points_will_shfted, width_min, width_max, height_min, height_max, H, W, pad_num):
    # points_will_shfted shape: (B, N, 2)
    padding = (int(abs(width_min)),int( abs(width_max - W)), int(abs(height_min)), int(abs(height_max - H)))
    # padding order is (left, right, top, bottom)
    # Shift the point pairs to match new resolution
    shifted_points = points_will_shfted.clone()
    
    shifted_points[:,:,0] = shifted_points[:,:,0] + padding[0]
    shifted_points[:,:,1] = shifted_points[:,:,1] + padding[2]
    return shifted_points

def boundary_src_and_tgt(points_src, points_dst, target_points, out_height, out_width):
    B = points_src.shape[0]
    # Create a mask where all conditions are met for each point
    mask = ((points_dst[:,:,0] >= 0) & 
            (points_dst[:,:,0] < out_width) & 
            (points_dst[:,:,1] >= 0) & 
            (points_dst[:,:,1] < out_height))
    mask2 = ((points_src[:,:,0] >= 0) &
            (points_src[:,:,0] < out_width) &
            (points_src[:,:,1] >= 0) &
            (points_src[:,:,1] < out_height))
    mask = mask & mask2
    
    # # Use this mask to filter target_points
    mask = mask.unsqueeze(-1).expand_as(target_points)
    points_dst = points_dst[mask].view(B, -1, 2)
    points_src = points_src[mask].view(B, -1, 2)
    return points_src, points_dst



def dilate_thin_area(mask, dilation_kernel_size=8, thickening_kernel_size=8, is_plot=True):
    _,origin_channel,H,W = mask.shape
    mask = mask[:,0:1,:,:]
    channels = mask.shape[1]
    kernel_size = (dilation_kernel_size, dilation_kernel_size)
    kernel = torch.ones((channels, 1, *kernel_size), device=mask.device, dtype=mask.dtype)
    
    # Perform morphological open (erosion followed by dilation)
    # For erosion, we want to keep only the areas where the 1's in the kernel completely overlap the 1's in the image
    erosion = F.conv2d(mask, kernel, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=channels)
    erosion = (erosion == kernel.numel()).float()
    # For dilation, we want to activate all the pixels where there's an overlap with the kernel
    dilation = F.conv2d(erosion, kernel, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=channels)
    dilation = (dilation >= 1).float()
    # Final image is the result of dilation
    dilated_eroded_mask = dilation
    # make dilated_eroded_mask same size as mask
    dilated_eroded_mask = dilated_eroded_mask[:, :, :H, :W]
    thick_area = mask * dilated_eroded_mask
    thick_area = thick_area.clamp(0, 1)


    thin_area = mask * (1 - thick_area)

    # dilate thin area
    kernel_size = (thickening_kernel_size, thickening_kernel_size)
    channels = thin_area.shape[1]
    kernel = torch.ones((channels, 1, *kernel_size), device=thin_area.device, dtype=thin_area.dtype)
    dilate_thin_area = F.conv2d(thin_area, kernel, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=channels)
    dilate_thin_area = (dilate_thin_area >= 1).float()
    dilate_thin_area = dilate_thin_area[:, :, :H, :W]
    # combine thick_area and dilate_thin_area
    result = thick_area + dilate_thin_area
    result = result.clamp(0, 1)
    result = result.repeat(1,origin_channel,1,1)

    return result


def dilate_mask(mask, kernel_size=3, iterations = 1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = np.array(to_pillow_fn(mask))
    dilation = cv2.dilate(image, kernel, iterations = iterations)
    
    dilation = dilation[..., 0] / 255.0
    dilation_tensor = torch.tensor( dilation).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device = mask.device, dtype=mask.dtype)
    
    return dilation_tensor