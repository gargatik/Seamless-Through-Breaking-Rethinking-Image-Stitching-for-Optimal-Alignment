import os
import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy
import skimage.io
import torch
import torch.nn.functional as F


# import Imath
# import OpenEXR
class Warper:
    def __init__(self, resolution: tuple = None, device = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        return

    def forward_warp(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                     transformation1: torch.Tensor, transformation2: torch.Tensor, intrinsic1: torch.Tensor, 
                     intrinsic2: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device)
        mask1 = mask1.to(self.device)
        depth1 = depth1.to(self.device)
        transformation1 = transformation1.to(self.device)
        transformation2 = transformation2.to(self.device)
        intrinsic1 = intrinsic1.to(self.device)
        intrinsic2 = intrinsic2.to(self.device)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2) # [1, H, W, 3, 1]
        # trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
        # trans_depth1 = trans_points1[:, :, 2, 0]
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0]

        grid = self.create_grid(b, h, w).to(trans_coordinates)

        trans_coordinates = trans_coordinates.permute(0,3,1,2) # (b, h, w, 2) --> (b, 2, h, w)

        # print("trans_coordinates",trans_coordinates.shape,"grid",grid.shape)
        flow12 = trans_coordinates - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)

        # print(" warped_frame2, mask2 ", warped_frame2.shape, mask2.shape )
        warped_depth2 = self.bilinear_splatting(trans_depth1[:, :, None], mask1, trans_depth1, flow12, None,
                                                is_image=False)[0]
        # print("warped_depth2",warped_depth2.shape)
        warped_depth2 = warped_depth2[:, :, 0]
        # print("warped_depth2",warped_depth2.shape)
        return warped_frame2, mask2, warped_depth2, flow12
    def forward_warp_with_flow(self, image: torch.Tensor, flow: torch.Tensor, is_range_0_255=False, is_just_shift=False,min_W_max_W_min_H_max_H=None, depth1=None, mask1=None, flow_mask=None, return_min_W_max_W_min_H_max_H=False):
        b, c, h, w = image.shape
        # assert image.shape == (b, 3, h, w)
        assert flow.shape == (b, 2, h, w)

        if is_range_0_255:
            # [0,255]to [-1, 1]
            image = image / 255.0 * 2.0 - 1.0
            
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(image)
        if depth1 is None:
            depth1 = torch.ones(size=(b, h, w)).to(image)
        
        result = self.bilinear_splatting(image, mask1, depth1, flow, flow_mask, is_image=True, is_just_shift=is_just_shift, min_W_max_W_min_H_max_H=min_W_max_W_min_H_max_H, return_min_W_max_W_min_H_max_H=return_min_W_max_W_min_H_max_H)
        warped_frame2, mask2 = result[0], result[1]    
        if is_range_0_255:
            #  [-1, 1] to [0,255]
            warped_frame2 = (warped_frame2 + 1.0) / 2.0 * 255.0
        if return_min_W_max_W_min_H_max_H:
            min_W_max_W_min_H_max_H = result[2]
            return warped_frame2, mask2, min_W_max_W_min_H_max_H
        return warped_frame2, mask2



    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False, is_just_shift=False,min_W_max_W_min_H_max_H=None, return_min_W_max_W_min_H_max_H=False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid
        print("trans_pos",trans_pos.shape, trans_pos.min(), trans_pos.max())




        # new h,w
        if min_W_max_W_min_H_max_H == None:
            maxH_W = max(h,w)
            trans_pos = trans_pos.clip(-2*maxH_W,2*maxH_W)
            print("trans_pos clip",trans_pos.shape, trans_pos.min(), trans_pos.max())
            max_H = max( trans_pos[:,1,:,:].max(), torch.tensor(h).to(trans_pos) * 1.0).ceil().item()
            max_W = max( trans_pos[:,0,:,:].max(), torch.tensor(w).to(trans_pos) * 1.0).ceil().item()
            min_H = min( trans_pos[:,1,:,:].min(), torch.tensor(0.0).to(trans_pos)).item()
            min_W = min( trans_pos[:,0,:,:].min(), torch.tensor(0.0).to(trans_pos)).item()
         
        else:
            min_W, max_W, min_H, max_H = min_W_max_W_min_H_max_H
        # to int
        max_H = int(max_H)
        max_W = int(max_W)
        min_H = int(min_H)
        min_W = int(min_W)
        print("min_H", min_H, "max_H", max_H, "min_W", min_W, "max_W", max_W)
        min_W_max_W_min_H_max_H = [min_W, max_W, min_H, max_H]
        target_H = int(max_H - min_H) 
        target_W = int(max_W - min_W) 
        print("target_H",target_H, "target_W",target_W)
        # H, W = flow12.shape[-2:]
        # vgrid, grid = flow_to_warp(flow12) # [B, H, W, 2] 
        # print("vgrid",vgrid.shape, vgrid.min(), vgrid.max())
        # maxH_W = max(H,W)
        # vgrid = vgrid.clip(-2*maxH_W,2*maxH_W)
        # print("vgrid clip",vgrid.shape, vgrid.min(), vgrid.max())
        # max_H = max( vgrid[:,:,:,1].max(), H)
        # max_W = max( vgrid[:,:,:,0].max(), W)
        # min_H = min( vgrid[:,:,:,1].min(), 0)
        # min_W = min( vgrid[:,:,:,0].min(), 0)
        # ## 0-H, 0-W --> min_H-max_H, min_W-max_W
        # target_H = int(max_H - min_H)
        # target_W = int(max_W - min_W)
        # print("target_H",target_H, "target_W",target_W)
        shift_WH = [-min_W, -min_H]
        print("shift_WH",shift_WH)

        grid = self.create_grid(b, h, w).to(frame1)
        shift_delta_flow = torch.tensor(shift_WH).reshape(1,2,1,1).repeat(1, 1, h, w ).to(grid)
        trans_pos = flow12 + grid + shift_delta_flow
        if is_just_shift:
            trans_pos = grid + shift_delta_flow
        
        h, w = target_H, target_W




        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)
        # depth_weights= torch.ones_like(depth_weights)
        print("depth_weights",depth_weights.min(),depth_weights.max())

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        print("warped_frame_cf",warped_frame_cf.shape)
        print("cropped_warped_frame",cropped_warped_frame.shape)

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            # assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            # assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        if return_min_W_max_W_min_H_max_H:
            return warped_frame2, mask2, min_W_max_W_min_H_max_H
        return warped_frame2, mask2

    def bilinear_interpolation(self, frame2: torch.Tensor, mask2: Optional[torch.Tensor], flow12: torch.Tensor,
                               flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear interpolation
        :param frame2: (b, c, h, w)
        :param mask2: (b, 1, h, w): 1 for known, 0 for unknown. Optional
        :param flow12: (b, 2, h, w)
        :param flow12_mask: (b, 1, h, w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame1: (b, c, h, w)
                 mask1: (b, 1, h, w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame2.shape[2:4] == self.resolution
        b, c, h, w = frame2.shape
        if mask2 is None:
            mask2 = torch.ones(size=(b, 1, h, w)).to(frame2)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame2)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        weight_nw = torch.moveaxis(prox_weight_nw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])

        frame2_offset = F.pad(frame2, [1, 1, 1, 1])
        mask2_offset = F.pad(mask2, [1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None]

        f2_nw = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        f2_sw = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        f2_ne = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        f2_se = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        m2_nw = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        m2_sw = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        m2_ne = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        m2_se = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        nr = weight_nw * f2_nw * m2_nw + weight_sw * f2_sw * m2_sw + \
             weight_ne * f2_ne * m2_ne + weight_se * f2_se * m2_se
        dr = weight_nw * m2_nw + weight_sw * m2_sw + weight_ne * m2_ne + weight_se * m2_se

        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=nr.dtype, device=nr.device)
        warped_frame1 = torch.where(dr > 0, nr / dr, zero_tensor)
        mask1 = (dr > 0).to(frame2)

        # Convert to channel first
        warped_frame1 = torch.moveaxis(warped_frame1, [0, 1, 2, 3], [0, 2, 3, 1])
        mask1 = torch.moveaxis(mask1, [0, 1, 2, 3], [0, 2, 3, 1])

        if is_image:
            assert warped_frame1.min() >= -1.1  # Allow for rounding errors
            assert warped_frame1.max() <= 1.1
            warped_frame1 = torch.clamp(warped_frame1, min=-1, max=1)
        
        return warped_frame1, mask1

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            raise NotImplementedError
            exr_file = OpenEXR.InputFile(path.as_posix())
            # raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        # check device is string or torch.device
        if isinstance(device, torch.device):
            return device

        if device == 'cpu':
            device = torch.device('cpu')
        elif  device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            raise NotImplementedError(f'Unknown device: {device}')
        return device
