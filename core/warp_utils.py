
import torch
import torch.nn.functional as F
from core.utils.utils import coords_grid
import torchvision.transforms as T
resize_512 = T.Resize((512,512))


""" HOMOGRAHPY MESH UTIL"""
def get_rigid_mesh(batch_size, height, width, grid_h = 511, grid_w = 511):
    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return ori_pt

def H2Mesh(H, rigid_mesh, grid_h=511, grid_w=511):
    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])
    return mesh


""" FLOW UTILS """
def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
    mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow
""" FLOW WARP UTIL """

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def flow_to_warp(flow):
    """
    Compute the warp from the flow field
    Args:
        flow: optical flow shape [B, 2, H, W]
    Returns:
        warp: the endpoints of the estimated flow. shape [B, H, W, 2]
    """
    flow = flow.permute(0, 2, 3, 1)
    B, H, W, _ = flow.size()
    grid = coords_grid(B, H, W)
    if flow.is_cuda:
        grid = grid.cuda()
    grid = grid.permute(0, 2, 3, 1)
    warp = grid + flow
    return warp

def warp(x, flo, mode='bilinear'):
    H, W = flo.shape[-2:]
    vgrid = flow_to_warp(flo)
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W-1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H-1, 1) - 1.0
    if mode == 'bilinear':
        output = F.grid_sample(x, vgrid, mode=mode, align_corners=True)
    else:
        output = F.grid_sample(x, vgrid, mode=mode)
    return output


""" COMPUTE OUCCLUSTION UTIL"""
def mask_invalid(coords, pad_h=0, pad_w=0):
    """
    Mask coordinates outside of the image.

    Valid = 1, invalid = 0.

    Args:
        coords: a 4D float tensor of image coordinates. [B, H, W, 2]
        pad_h: int, the amount of padding applied to the top of the image
        pad_w: int, the amount of padding applied to the left of the image

    Returns:
        The mask showing which coordinates are valid. [B, 1, H, W]
    """
    pad_h = float(pad_h)
    pad_w = float(pad_w)
    coords_rank = len(coords.shape)
    if coords_rank != 4:
        raise NotImplementedError()
    max_height = float(coords.shape[-3] - 1)
    max_width = float(coords.shape[-2] - 1)
    mask = torch.logical_and(
        torch.logical_and(coords[:, :, :, 0] >= pad_w,
                        coords[:, :, :, 0] <= max_width),
        torch.logical_and(coords[:, :, :, 1] >= pad_h,
                        coords[:, :, :, 1] <= max_height))
    mask = mask.float()[:, None, :, :]
    return mask


def compute_range_map(flow):
    """Using backward flow to compute the range map"""
    # permute flow from [B, 2, H, W] to shape [B, H, W, 2]
    batch_size, _, input_height, input_width = flow.size()

    coords = flow_to_warp(flow)

    # split coordinates into an integer part and a float offset for interpolation.
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.to(torch.int32)

    # Define a batch offset for flattened indexes into all pixels
    batch_range = torch.reshape(torch.arange(batch_size), [batch_size, 1, 1])
    if flow.is_cuda:
        batch_range = batch_range.cuda()
    idx_batch_offset = batch_range.repeat(1, input_height, input_width) * input_height * input_width

    # Flatten everything
    coords_floor_flattened = coords_floor.reshape(-1, 2)
    coords_offset_flattened = coords_offset.reshape(-1, 2)
    idx_batch_offset_flattened = idx_batch_offset.reshape(-1)

    # Initialize results
    idxs_list = []
    weights_list = []

    # Loop over different di and dj to the four neighboring pixels
    for di in range(2):
        for dj in range(2):
            # Compute the neighboring pixel coordinates
            idxs_i = coords_floor_flattened[:, 0] + di
            idxs_j = coords_floor_flattened[:, 1] + dj
            # Compute the flat index into all pixels
            idxs = idx_batch_offset_flattened + idxs_j * input_width + idxs_i

            # Only count valid pixels
            mask = torch.nonzero(torch.logical_and(
                torch.logical_and(idxs_i >= 0, idxs_i < input_width),
                torch.logical_and(idxs_j >= 0, idxs_j < input_height)
            ), as_tuple=True)
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flattened[mask]

            # Compute weights according to bilinear interpolation
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            # Append indices and weights
            idxs_list.append(valid_idxs)
            weights_list.append(weights)

    # Concatenate everything
    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    counts = torch.zeros(batch_size * input_width * input_height, dtype=weights.dtype)
    if flow.is_cuda:
        counts = counts.cuda()
    counts.scatter_add_(0, idxs, weights)
    range_map = counts.reshape(batch_size, 1, input_height, input_width)
    return range_map

def compute_fb_consistency(flow_ij, flow_ji):
    # Compare forward and backward flow
    flow_ji_in_i = warp(flow_ji, flow_ij)
    fb_sq_diff = torch.sum((flow_ij + flow_ji_in_i)**2, dim=1, keepdim=True)
    fb_sum_sq = torch.sum((flow_ij**2 + flow_ji_in_i**2), dim=1, keepdim=True)

    return fb_sq_diff, fb_sum_sq

def compute_occlusion(flow_ij, flow_ji, occlusion_estimation, occlusion_are_zeros=False, boundaries_occluded=True):
    """
    Compute occlusion mask.
    Args:
        flow_ij: the forward optical flow [B, 2, H, W]
        flow_ji: the backward optical flow [B, 2, H, W]
        occlusion_estimation: the algorithm chosed to estimate the occlusion. ('brox', 'fb_abs', 'wang')
        occlusion_are_zeros: the occluded regions are 0 (default false)
        boundaries_occluded: If True, treat flow vectors pointing off the boundaries
        as occluded. Otherwise explicitly mark them as unoccluded.
    Return:
        occlusion_mask: the estimated occlusion mask [B, 1, H, W]
    """

    # Compare forward and backward flow
    fb_sq_diff, fb_sum_sq = compute_fb_consistency(flow_ij, flow_ji)

    occlusion_mask = torch.zeros_like(flow_ij[:, :1, :, :])
    
    if occlusion_estimation == 'none':
        B, _, H, W = flow_ij.shape
        occlusion_mask = torch.zeros(B, 1, H, W, dtype=flow_ij.dtype, device=flow_ij.device)
    elif occlusion_estimation == 'brox':
        occlusion_mask = (fb_sq_diff > 0.01 * fb_sum_sq + 0.5).float()
    elif occlusion_estimation == 'fb_abs':
        occlusion_mask = (fb_sq_diff**0.5 > 1.5).float()
    elif occlusion_estimation == 'wang':
        range_map = compute_range_map(flow_ji)
        occlusion_mask = 1 - torch.clamp(range_map, min=0.0, max=1.0)
    
    if not boundaries_occluded:
        warp = flow_to_warp(flow_ij)
        occlusion_mask = torch.min(occlusion_mask, mask_invalid(warp))

    if occlusion_are_zeros:
        occlusion_mask = 1 - occlusion_mask
    return occlusion_mask
