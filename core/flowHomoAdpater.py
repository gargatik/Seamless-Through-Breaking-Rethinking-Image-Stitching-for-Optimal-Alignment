
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.udis_utils.torch_DLT as torch_DLT
import core.udis_utils.torch_homo_transform as torch_homo_transform

from core.utils.warper import Warper

import numpy as np
import cv2

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

from core.warp_utils import *

def preprocess_occlusion_mask(occlusion_mask, kernel_size = (19, 19)):
    occlusion_mask = (occlusion_mask >= 0.5).float()
    # The kernel's shape should match the 'occlusion_mask' shape [B, C, H, W]
    batch_size, channels, height, width = occlusion_mask.shape
    kernel = torch.ones((channels, 1, *kernel_size), device=occlusion_mask.device, dtype=occlusion_mask.dtype)
    # Perform morphological open (erosion followed by dilation)
    # For erosion, we want to keep only the areas where the 1's in the kernel completely overlap the 1's in the image
    erosion = F.conv2d(occlusion_mask, kernel, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=channels)
    erosion = (erosion == kernel.numel()).float()
    # For dilation, we want to activate all the pixels where there's an overlap with the kernel
    dilation = F.conv2d(erosion, kernel, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=channels)
    dilation = (dilation >= 1).float()
    # Final image is the result of dilation
    dilated_eroded_image = dilation
    # Threshold to make sure everything is 0 or 1
    occlusion_mask = (dilated_eroded_image >= 0.5).float()
    # Returning the processed mask
    return occlusion_mask

# define and forward
class FlowHomoAdpater(nn.Module):

    def __init__(self, homo_backbone, flow_backbone, cfg):
        super(FlowHomoAdpater, self).__init__()
        self.cfg = cfg
        self.use_forward = cfg.use_forward if hasattr(cfg, "use_forward") else False
        print("use_foward", self.use_forward)
        self.detach_H = cfg.detach_H if hasattr(cfg,"detach_H") else False
        print("detach_H", self.detach_H)
        self.detach_flow = cfg.detach_flow if hasattr(cfg, "detach_flow") else False
        print("detach_flow", self.detach_flow)

        self.homo_backbone = homo_backbone
        self.flow_backbone = flow_backbone
      
    def predict_homo(self, input1_tensor, input2_tensor):
        # [0,255] normalize to [-1, 1]
        input1 = (input1_tensor / 127.5) - 1.0
        input2 = (input2_tensor / 127.5) - 1.0
        offset_1, _ = self.homo_backbone(input1, input2)
        H_motion_1 = offset_1.reshape(-1, 4, 2) # 4, 4, 2
        if self.detach_H:
            H_motion_1 = H_motion_1.detach()
        return H_motion_1
    
    def predict_flow(self, input1_tensor, input2_tensor): # predict flow12
        ouput_dict = {}
        flow_predictions = self.flow_backbone(input1_tensor, input2_tensor, ouput_dict)
        if self.flow_backbone.training:
            pass
        else:
            flow_predictions = [flow_predictions[0]]
        return flow_predictions
    
    def forward(self, input1_tensor, input2_tensor, type="train", pad_mode="constant",preprocess_callback=None): # [0,255]
        if type=="test_out":
            return self.test_out_forward(input1_tensor, input2_tensor, pad_mode=pad_mode,preprocess_callback=preprocess_callback)

        if type=="train" or type=="test_eval":
            return self.train_eval_foward(input1_tensor, input2_tensor)
        
        else:
            raise NotImplementedError
            
        
    def train_eval_foward(self, input1_tensor, input2_tensor):
        out_dict = {}
            
        batch_size, _, img_h, img_w = input1_tensor.size()
        
        # Homo
        H_motion_1 = self.predict_homo(input1_tensor, input2_tensor)
        ## 4 corner point
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8) # 4, 3, 3

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                    [0., img_h/8 / 2.0, img_h/8 / 2.0],
                    [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        mask = torch.ones_like(input2_tensor)
        output_H = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (int(img_h), int(img_w)))
        H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
        output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat, (img_h, img_w))

        if hasattr(self.cfg, "only_homo") and self.cfg.only_homo:
            final_warp_output = output_H #torch.zeros_like(output_H).to(torch.uint8)
            flow_predictions = None
            overlap = None
        # Flow 
        else:
            if self.use_forward:
                if hasattr(self.cfg, "use_fb_consistency_mask") and self.cfg.use_fb_consistency_mask:
                    raise NotImplementedError
                warper = Warper(device=input2_tensor.device)

                flow_predictions = self.predict_flow(input2_tensor, input1_tensor)
                predict_ = flow_predictions[-1]
                final_flow = H_flow + residual_flow
                
                min_W_max_W_min_H_max_H = (0, img_w, 0, img_h)
                final_warp_output, _ = warper.forward_warp_with_flow(torch.cat((input2_tensor, mask), 1), final_flow, is_range_0_255=True, is_just_shift=False, min_W_max_W_min_H_max_H=min_W_max_W_min_H_max_H)

                overlap = final_warp_output[:,3:6,...].mean(dim=1)
                overlap_one = torch.ones_like(overlap)
                overlap_zero = torch.zeros_like(overlap)
                overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)

                overlap = overlap_one

                del warper
                
                
            else:
                if hasattr(self.cfg, "use_combine_h_flow") and self.cfg.use_combine_h_flow:
                    if hasattr(self.cfg, "use_fb_consistency_mask") and self.cfg.use_fb_consistency_mask:
                        raise NotImplementedError
                    warp_input2_tensor, warp_input2_mask = output_H[:,0:3,...], output_H[:,3:6,...]
                    flow_predictions = self.predict_flow(input1_tensor, warp_input2_tensor)
                    
                    H = torch.inverse(H)
                    rigid_mesh = get_rigid_mesh(batch_size=batch_size, height=img_h, width=img_w, grid_h = img_h-1, grid_w =img_w-1).to(H.device)
                    H_mesh = H2Mesh(H, rigid_mesh, grid_h = img_h-1, grid_w =img_w-1)
                    H_flow = (H_mesh - rigid_mesh).permute(0,3,1,2) # B,2,H,W
                    residual_flow = flow_predictions[-1]
                    final_flow = H_flow + residual_flow

                    mask = torch.ones_like(input2_tensor)
                    final_warp_output = warp(torch.cat((input2_tensor, mask), 1), final_flow) 

                    overlap = final_warp_output[:,3:6,...].mean(dim=1)
                    overlap_one = torch.ones_like(overlap)
                    overlap_zero = torch.zeros_like(overlap)
                    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)
                
                else:
                    warp_input2_tensor, warp_input2_mask = output_H[:,0:3,...], output_H[:,3:6,...]
                    flow_predictions = self.predict_flow(input1_tensor, warp_input2_tensor)


                    final_warp_output = warp(output_H, flow_predictions[-1]) 
                    overlap = final_warp_output[:,3:6,...].mean(dim=1)
                    overlap_one = torch.ones_like(overlap)
                    overlap_zero = torch.zeros_like(overlap)
                    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)

                    if hasattr(self.cfg, "use_fb_consistency_mask") and self.cfg.use_fb_consistency_mask:
                        flow_ij = flow_predictions[-1]
                        flow_ji = self.predict_flow(warp_input2_tensor, input1_tensor)[-1].detach()
                        
                        occlusion_mask = compute_occlusion(flow_ij=flow_ij, flow_ji=flow_ji, occlusion_estimation="wang", occlusion_are_zeros=True, boundaries_occluded=True)
                        occlusion_mask = torch.where(occlusion_mask >= 0.5, torch.ones_like(occlusion_mask), torch.zeros_like(occlusion_mask))
                        final_warp_output = final_warp_output * occlusion_mask
                        occlusion_mask = occlusion_mask.squeeze(1)
                        
                        out_dict.update(origin_occlusion_mask = occlusion_mask )
                        

        out_dict.update(output_H=output_H, output_H_inv = output_H_inv, final_warp_output = final_warp_output, overlap = overlap, flow_predictions=flow_predictions, H=H)
        

        return out_dict

  
        
    
    
    def test_out_forward(self, input1_tensor, input2_tensor, pad_mode="constant", preprocess_callback = None): # [0,255]
        
        
        with torch.no_grad():
         
            batch_size, _, img_h, img_w = input1_tensor.size()
            ###### Process with 512 x 512 resolution #######
            input1_tensor_512 = resize_512(input1_tensor)
            input2_tensor_512 = resize_512(input2_tensor)
            # Homo
            H_motion_1 = self.predict_homo(input1_tensor_512, input2_tensor_512)
            # initialize the source points bs x 4 x 2
            src_p = torch.tensor([[0., 0.], [512, 0.], [0., 512], [512, 512]])
            if torch.cuda.is_available():
                src_p = src_p.cuda()
            src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
            # target points
            dst_p = src_p + H_motion_1
            # solve homo using DLT
            H = torch_DLT.tensor_DLT(src_p, dst_p)
            # scale to 512 
            M_tensor = torch.tensor([[512 / 2.0, 0., 512 / 2.0],
                            [0., 512 / 2.0, 512 / 2.0],
                            [0., 0., 1.]])
            if torch.cuda.is_available():
                M_tensor = M_tensor.cuda()
            M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            M_tensor_inv = torch.inverse(M_tensor)
            M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
            H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile) 
           
            img_h_512, img_w_512 = 512, 512
            mask = torch.ones_like(input2_tensor_512)
            output_H = torch_homo_transform.transformer(torch.cat((input2_tensor_512, mask), 1), H_mat, (int(img_h_512), int(img_w_512)))

            warp_input2_tensor_512, warp_input2_mask_512 = output_H[:,0:3,...], output_H[:,3:6,...]
            warp_input2_mask_512 = warp_input2_mask_512.mean(dim=1,keepdim=True)
            warp_input2_mask_512 = (warp_input2_mask_512 > 0.5).to(warp_input2_mask_512.dtype)

            flow_predictions_512 = self.predict_flow(input1_tensor_512, warp_input2_tensor_512)
            
            ######################################
            # scale all to origin resolution
            # scale flow to img_h, img_w
            flow_predictions = [resize_flow(flow, new_shape=(img_h, img_w)) for flow in flow_predictions_512 ]

            # scale H_motion to img_h, img_w
            H_motion = torch.stack([H_motion_1[...,0]*img_w/512, H_motion_1[...,1]*img_h/512], 2)
            # initialize the source points bs x 4 x 2
            src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
            if torch.cuda.is_available():
                src_p = src_p.cuda()
            src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
            # target points
            dst_p = src_p + H_motion
            # solve homo using DLT
            H = torch_DLT.tensor_DLT(src_p, dst_p)
            rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
            ini_mesh = H2Mesh(H, rigid_mesh)
            mesh = ini_mesh
            ######################################
            # Define the stitching resolution
            width_max = torch.max(mesh[...,0])
            width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
            width_min = torch.min(mesh[...,0])
            width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
            height_max = torch.max(mesh[...,1])
            height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
            height_min = torch.min(mesh[...,1])
            height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

            width_min, width_max, height_min, height_max  = width_min.int(), width_max.int(), height_min.int(), height_max.int()

            out_width = width_max - width_min
            out_height = height_max - height_min
            ######################################
            # get warped img1
            M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                            [0., out_height / 2.0, out_height / 2.0],
                            [0., 0., 1.]])
            N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                            [0., img_h / 2.0, img_h / 2.0],
                            [0., 0., 1.]])
            if torch.cuda.is_available():
                M_tensor = M_tensor.cuda()
                N_tensor = N_tensor.cuda()
            N_tensor_inv = torch.inverse(N_tensor)
            I_ = torch.tensor([[1., 0., width_min],
                            [0., 1., height_min],
                            [0., 0., 1.]])#.unsqueeze(0)
            mask = torch.ones_like(input1_tensor)
            if torch.cuda.is_available():
                I_ = I_.cuda()
                mask = mask.cuda()
            I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)
            homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), I_mat, (out_height.int(), out_width.int()))
            ######################################
            # get warped img2
            if self.use_forward:
                raise NotImplementError
            # backward
            else:          
                if hasattr(self.cfg, "test_not_use_combine_h_flow") and self.cfg.test_not_use_combine_h_flow:
                    if hasattr(self.cfg, "use_whole_resolution") and self.cfg.use_whole_resolution:
                        raise NotImplementError
                    
                    # get H warp2
                    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                    N_tile_inv = N_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
                    H = torch.matmul(H, I_.unsqueeze(0))
                    H_mat = torch.matmul(torch.matmul(N_tile_inv, H), M_tile)

                    mask = torch.ones_like(input2_tensor)
                    homo_output2 = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (out_height.int(), out_width.int()))
                    # get flow warp2
                    residual_flow = flow_predictions[-1]
                    flow_mask = torch.ones_like(residual_flow).mean(dim=1, keepdim=True)
                    residual_flow_output = torch_homo_transform.transformer(torch.cat((residual_flow, flow_mask), 1), I_mat, (out_height.int(), out_width.int()))
                    residual_flow_new_size, flow_mask_new_size = residual_flow_output[:,0:2,...], residual_flow_output[:,2:3,...] 
                    final_warp_output = warp(homo_output2, residual_flow_new_size)
                    final_warp_output = final_warp_output * flow_mask_new_size

                    occlusion_mask = None
                    origin_occlusion_mask = None
                    if hasattr(self.cfg, "use_fb_consistency_mask") and self.cfg.use_fb_consistency_mask:
                        # foward flow
                        foward_flow = residual_flow
                        
                        # back flow
                        flow_predictions_512 = self.predict_flow( warp_input2_tensor_512, input1_tensor_512)
                        ## scale all the flow in flow_prediction_512 to img_h, img_w
                        flow_predictions = [resize_flow(flow, new_shape=(img_h, img_w)) for flow in flow_predictions_512 ]
                        back_flow = flow_predictions[-1]

                        # compute occlusion and preprocess
                        occlusion_mask = compute_occlusion(flow_ij=foward_flow, flow_ji=back_flow, occlusion_estimation="wang", occlusion_are_zeros=True, boundaries_occluded=True)
                        occlusion_mask = preprocess_occlusion_mask(occlusion_mask)
                        origin_occlusion_mask = occlusion_mask.clone()
                        occlusion_mask = torch_homo_transform.transformer(occlusion_mask, I_mat, (out_height.int(), out_width.int()))
                        occlusion_mask = preprocess_occlusion_mask(occlusion_mask)

                        
                        final_warp_output  = final_warp_output * occlusion_mask
                        output1 , mask1 = homo_output[:,0:3,...], homo_output[:,3:6,...]
                        output2 , mask2 = final_warp_output[:,0:3,...], final_warp_output[:,3:6,...] 
                        
                        non_overlap_mask = 1 - mask1
                        # occlusion_mask AND non_overlap_mask (non_overlap_mask + occlusion_mask).clip(0,1)
                        output2 =  homo_output2[:,0:3,...] * (1-mask2) * non_overlap_mask + output2 * (mask2) 
                        mask2 = homo_output2[:,3:6,...] * (1-mask2) * non_overlap_mask + mask2 * mask2
                    else:
                        output1 , mask1 = homo_output[:,0:3,...], homo_output[:,3:6,...]
                        output2 , mask2 = final_warp_output[:,0:3,...], final_warp_output[:,3:6,...]
                        output2 = homo_output2[:,0:3,...] * (1-mask2) + output2 * (mask2)
                        mask2 = homo_output2[:,3:6,...] * (1-mask2) + mask2 * mask2
                        


                    blend_image = (output1 * mask1 + output2 * mask2) / (mask1 + mask2)
                    blend_image = blend_image.clip(0,255).to(torch.uint8)
                    

                    mask1 = mask1.mean(dim=1,keepdim=True).clip(0,1).repeat(1,3,1,1)
                    mask2 = mask2.mean(dim=1,keepdim=True).clip(0,1).repeat(1,3,1,1)

                else:
                    raise NotImplementError

            out_dict = {}
            out_dict.update(H_warp=homo_output2[:,0:3,...], final_warp=final_warp_output[:,0:3,...] ,output1=output1, output2=output2, mask1=mask1, mask2=mask2, blend_image=blend_image, residual_flow= residual_flow)
            out_dict.update(width_min=width_min.item(),height_min=height_min.item(),out_height=out_height.item(), out_width=out_width.item())
            out_dict.update(H=H)
            out_dict.update(warp_input2_mask=warp_input2_mask_512)
            out_dict.update(warp_input2_tensor_512=warp_input2_tensor_512)
            out_dict.update(I_mat=I_mat)
            
            out_dict.update(H_warp_mask = homo_output2[:,3:6,...])
            if hasattr(self.cfg, "use_fb_consistency_mask") and self.cfg.use_fb_consistency_mask:
                out_dict.update(occlusion_mask=occlusion_mask)
                out_dict.update(origin_occlusion_mask=origin_occlusion_mask)
            return out_dict

    


    