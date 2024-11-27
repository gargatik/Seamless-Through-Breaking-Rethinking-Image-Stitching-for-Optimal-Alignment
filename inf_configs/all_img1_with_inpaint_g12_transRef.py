from yacs.config import CfgNode as CN
# TPS_PIPLINE CONFIG 
def get_tps_pipline_config(cfg):
    TPS_PIPELINE_CONFIG = CN()
    """ Revice the following parameters to get the desired results """
    # Options: 
    # - inpainter (based on diffusion, high quality, low speed)
    # - transref_inpainter (high speed, low quality)
    TPS_PIPELINE_CONFIG.inpainter = "transref_inpainter" 
    
    # Options:  
    # - all_img1_with_inpaint : use source image to fill most of the holes, remain little inpaint by inpainter
    # - inpaint_all_area : use inpainter to fill all holes 
    TPS_PIPELINE_CONFIG.mix_method =  "all_img1_with_inpaint" 
    
    # Options: 
    # - 8,8
    # - 12,12
    # - 16,16
    TPS_PIPELINE_CONFIG.grid_h, TPS_PIPELINE_CONFIG.grid_w = 12,12

    """ ---------------------------------------------------------- """

    TPS_PIPELINE_CONFIG.get_pt_methods = ["advanced_uniform_multi"]
    TPS_PIPELINE_CONFIG.tps_method="opencv" # Options: "opencv", "kornia", "other"

    TPS_PIPELINE_CONFIG.is_plot=False
    TPS_PIPELINE_CONFIG.limit_border_value = False 
    TPS_PIPELINE_CONFIG.inpaint_flow = False
    TPS_PIPELINE_CONFIG.inpaint_img = True
    TPS_PIPELINE_CONFIG.flow_pad_mode = "replicate"
    TPS_PIPELINE_CONFIG.mesh_pad_mode = None 

    TPS_PIPELINE_CONFIG.pad_num = 4
    TPS_PIPELINE_CONFIG.add_corner= False
    TPS_PIPELINE_CONFIG.flow_limit = -1
    TPS_PIPELINE_CONFIG.use_valid_on_flow = False
    
    TPS_PIPELINE_CONFIG.add_meshgrid = False
    TPS_PIPELINE_CONFIG.affine_scale=1.0
    TPS_PIPELINE_CONFIG.kernel_scale=1.0 #0.1
    TPS_PIPELINE_CONFIG.use_boundary_limit= False 
    TPS_PIPELINE_CONFIG.residual_flow_use_forward = cfg.use_foward

    TPS_PIPELINE_CONFIG.use_occ_filter = True
    TPS_PIPELINE_CONFIG.use_border_points_mask = True
    TPS_PIPELINE_CONFIG.do_avg_pooling = True

    TPS_PIPELINE_CONFIG.occlusion_mask= None
    TPS_PIPELINE_CONFIG.use_composition_when_inpaint= False

    TPS_PIPELINE_CONFIG.output2_is_only_tps = True
    assert TPS_PIPELINE_CONFIG.output2_is_only_tps == True
    TPS_PIPELINE_CONFIG.resize_to_area_limit_before_inpaint = 750*750

    return TPS_PIPELINE_CONFIG

#INFERENCE CONFIG [CONSTANT]
def get_infernce_config():
    INFERENCE_CONFIG = CN()
    INFERENCE_CONFIG.is_plot=False
    INFERENCE_CONFIG.eval = "udis_eval"
    INFERENCE_CONFIG.only_init_model = False
    INFERENCE_CONFIG.use_composition = True
    INFERENCE_CONFIG.composition_model_path = "./core/UDIS2/Composition/pretrained_model/epoch050_model.pth"
    INFERENCE_CONFIG.resize_to_512 = False
    INFERENCE_CONFIG.pad_mode = "replicate"
    INFERENCE_CONFIG.restore_ckpt = ""
    INFERENCE_CONFIG.test_not_use_combine_h_flow = True
    INFERENCE_CONFIG.swap_image = False
    INFERENCE_CONFIG.use_forward = False 
    INFERENCE_CONFIG.use_fb_consistency_mask= True 
    INFERENCE_CONFIG.use_whole_resolution = False
    return INFERENCE_CONFIG


    