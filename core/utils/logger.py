import wandb
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger


from core.warp_utils import warp

class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg
        self.use_wandb = None if "use_wandb" not in self.cfg  else self.cfg.use_wandb 
        if self.use_wandb:
            wandb_mode = self.cfg.wandb_mode if hasattr(self.cfg, "wandb_mode") else "online"
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_name, config=self.cfg, mode=wandb_mode)
            if hasattr(self.cfg, "wandb_use_watch_model") and self.cfg.wandb_use_watch_model:
                print("watch model")
                wandb.watch(self.model, log="all")
            

    def _print_training_status(self):
        # metrics_data = [self.running_loss[k]/self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
       
        training_str = "[{:6d}, {}] ".format(self.total_steps+1, self.scheduler.get_last_lr())
        metrics_str = ""
        for k in sorted(self.running_loss.keys()):
            metrics_str += "{}: {:10.4f}, ".format(k, self.running_loss[k]/self.cfg.sum_freq)
        
        
        # print the training status
        loguru_logger.info(training_str + metrics_str)

        # if self.writer is None:
        #     if self.cfg.log_dir is None:
        #         self.writer = SummaryWriter()
        #     else:
        #         self.writer = SummaryWriter(self.cfg.log_dir)

        for k in self.running_loss:
            # self.writer.add_scalar(k, self.running_loss[k]/self.cfg.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]
            
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)

        if self.total_steps == 1 or (self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq-1):
            self._print_training_status()
            self.running_loss = {}
    
    @torch.no_grad()
    def plot_img_dicts(self, img_dict, phase):
        fig, axs = plt.subplots(1, len(img_dict.keys()), figsize=(15, 15))
        # axs[0].imshow(img_dict["image1"])
        # axs[0].set_title('image1')

        # axs[1].imshow(img_dict["image2"])
        # axs[1].set_title('image2')

        # axs[2].imshow(img_dict["warp_H"])
        # axs[2].set_title('warp_H (warp image2)')
        
        # axs[3].imshow(img_dict["final_warp"])
        # axs[3].set_title('final warp image2')

        # axs[4].imshow(img_dict["mask2"])
        # axs[4].set_title('mask2')

        # axs[5].imshow(img_dict["overlap"])
        # axs[5].set_title('overlap')
        for idx, (key, value) in enumerate(img_dict.items()):
            axs[idx].imshow(value)
            axs[idx].set_title(key)

        save_path = os.path.join(self.cfg.log_dir, f"{phase}_result{self.cfg.suffix}.png")
        plt.savefig(save_path)
        print("save file to", save_path)

        if self.use_wandb:
            wandb.log({
                f"{phase}_result": wandb.Image(save_path),
            }, step=self.total_steps)
        save_path_dict = {
            f"{phase}_result": save_path,
        }

        return save_path_dict


    
    
    @torch.no_grad()
    def plot_the_results(self, data_dict, result_dict, phase):
        image1 = data_dict['image1']
        image2 = data_dict['image2']

        flow = data_dict['flow']
        valid = data_dict['valid']
        predict_flow = result_dict['predict_flow']
        warped_image_gt = warp(image2, flow)  *  valid.unsqueeze(1)
        warped_image_pred = warp(image2, predict_flow) *  valid.unsqueeze(1)

        
        

        # print("image1", image1.shape, "flow", flow.shape, "valid", valid.shape)
        # print("image1", image1.min(),image1.max(), "flow", flow.min(),flow.max(), "valid", valid.min(),valid.max()) 
        # print("predict_flow", predict_flow.shape, "warped_image_gt", warped_image_gt.shape, "warped_image_pred", warped_image_pred.shape)

        # image1 torch.Size([2, 3, 512, 640]) flow torch.Size([2, 2, 512, 640]) valid torch.Size([2, 512, 640])
        # predict_flow torch.Size([2, 2, 512, 640]) warped_image_gt torch.Size([2, 3, 512, 640]) warped_image_pred torch.Size([2, 3, 512, 640])
        


        # plot as grids
        # image1, image2, valid
        # flow x, warped_image_gt (warp 2), flow y
        # predict_flow x , warped_image_pred (warp2), predict_flow y
        sample_idx = 0
        
        image1 = image1[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8)
        # print("image1",image1.shape,image1.min(),image1.max())

        image2 = image2[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8)

        valid = valid[sample_idx].detach().cpu()

        flow_x = flow[sample_idx,0,:,:].detach().cpu()
        flow_y = flow[sample_idx,1,:,:].detach().cpu()

        warped_image_gt = warped_image_gt[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8)
        avg_blend_image_gt = ((image1.float() + warped_image_gt.float()) / 2).clip(0,255).to(torch.uint8)
        # print("warped_image_gt",warped_image_gt.shape,warped_image_gt.min(),warped_image_gt.max())


        predict_flow_x = predict_flow[sample_idx,0,:,:].detach().cpu()
        predict_flow_y = predict_flow[sample_idx,1,:,:].detach().cpu()

        warped_image_pred = warped_image_pred[sample_idx].detach().cpu().permute(1,2,0).clip(0,255).to(torch.uint8)
        avg_blend_image_pred = ((image1.float() + warped_image_pred.float()) / 2).clip(0,255).to(torch.uint8)
        # print("warped_image_pred",warped_image_pred.shape,warped_image_pred.min(),warped_image_pred.max())


        visual_max_flow_threshold = 200 # self.cfg.max_flow
        visual_min_flow_threshold = -200
        visual_min_flow_fn = lambda x : max(visual_min_flow_threshold, x.min())
        visual_max_flow_fn = lambda x : min(visual_max_flow_threshold, x.max())

        # plot as grids
        # image1, image2, valid, 
        # flow x, flow y, warped_image_gt (warp 2), avg_blend_image_gt
        # predict_flow x , predict_flow y, warped_image_pred (warp2), avg_blend_image_pred
        fig, axs = plt.subplots(3, 4, figsize=(15, 15))
        axs[0, 0].imshow(image1)
        axs[0, 0].set_title('image1')

        axs[0, 1].imshow(image2)
        axs[0, 1].set_title('image2')

        axs[0, 2].imshow(valid, cmap='gray', vmin=0, vmax=1)
        axs[0, 2].set_title('valid')

        axs[1, 0].imshow(flow_x, vmin=visual_min_flow_fn(flow_x) , vmax=visual_max_flow_fn(flow_x))
        axs[1, 0].set_title(f'flow x, min:{flow_x.min():.2f}, max:{flow_x.max():.2f}')

        axs[1, 1].imshow(flow_y, vmin=visual_min_flow_fn(flow_y) , vmax=visual_max_flow_fn(flow_y))
        axs[1, 1].set_title(f'flow y, min:{flow_y.min():.2f}, max:{flow_y.max():.2f}')

        axs[1, 2].imshow(warped_image_gt)
        axs[1, 2].set_title('warped_image_gt (warp image2)')

        axs[1, 3].imshow(avg_blend_image_gt)
        axs[1, 3].set_title('avg_blend_image_gt')

        
        axs[2, 0].imshow(predict_flow_x, vmin=visual_min_flow_fn(predict_flow_x) , vmax=visual_max_flow_fn(predict_flow_x))
        axs[2, 0].set_title(f'predict_flow x, min:{predict_flow_x.min():.2f}, max:{predict_flow_x.max():.2f}')

        axs[2, 1].imshow(predict_flow_y, vmin=visual_min_flow_fn(predict_flow_y) , vmax=visual_max_flow_fn(predict_flow_y))
        axs[2, 1].set_title(f'predict_flow y, min:{predict_flow_y.min():.2f}, max:{predict_flow_y.max():.2f}')

        axs[2, 2].imshow(warped_image_pred)
        axs[2, 2].set_title('warped_image_pred (warp image2)')

        axs[2, 3].imshow(avg_blend_image_pred)
        axs[2, 3].set_title('avg_blend_image_pred')
        
        save_path = os.path.join(self.cfg.log_dir, f"{phase}_result{self.cfg.suffix}.png")
        plt.savefig(save_path)
        print("save file to", save_path)

        plt.clf()
        plt.imshow(flow_x, vmin=visual_min_flow_fn(flow_x) , vmax=visual_max_flow_fn(flow_x))
        plt.title(f'flow x, min:{flow_x.min():.2f}, max:{flow_x.max():.2f}')
        plt.colorbar()
        flow_x_save_path = os.path.join(self.cfg.log_dir, f"{phase}_flow_x{self.cfg.suffix}.png")
        plt.savefig(flow_x_save_path)

        plt.clf()
        plt.imshow(flow_y, vmin=visual_min_flow_fn(flow_y) , vmax=visual_max_flow_fn(flow_y))
        plt.title(f'flow y, min:{flow_y.min():.2f}, max:{flow_y.max():.2f}')
        plt.colorbar()
        flow_y_save_path = os.path.join(self.cfg.log_dir, f"{phase}_flow_y{self.cfg.suffix}.png")
        plt.savefig(flow_y_save_path)

        plt.clf()
        plt.imshow(predict_flow_x, vmin=visual_min_flow_fn(predict_flow_x) , vmax=visual_max_flow_fn(predict_flow_x))
        plt.title(f'predict_flow x, min:{predict_flow_x.min():.2f}, max:{predict_flow_x.max():.2f}')
        plt.colorbar()
        predict_flow_x_save_path = os.path.join(self.cfg.log_dir, f"{phase}_predict_flow_x{self.cfg.suffix}.png")
        plt.savefig(predict_flow_x_save_path)

        plt.clf()
        plt.imshow(predict_flow_y, vmin=visual_min_flow_fn(predict_flow_y) , vmax=visual_max_flow_fn(predict_flow_y))
        plt.title(f'predict_flow y, min:{predict_flow_y.min():.2f}, max:{predict_flow_y.max():.2f}')
        plt.colorbar()
        predict_flow_y_save_path = os.path.join(self.cfg.log_dir, f"{phase}_predict_flow_y{self.cfg.suffix}.png")
        plt.savefig(predict_flow_y_save_path)


        
        plt.clf()
        plt.imshow(avg_blend_image_pred)
        plt.title("avg_blend_image_pred")
        avg_blend_image_pred_save_path = os.path.join(self.cfg.log_dir, f"{phase}_avg_blend_image_pred{self.cfg.suffix}.png")
        plt.savefig(avg_blend_image_pred_save_path)


        
        save_path_dict = {
            f"{phase}_result": save_path,
            f"{phase}_flow_x": flow_x_save_path,
            f"{phase}_flow_y": flow_y_save_path,
            f"{phase}_predict_flow_x": predict_flow_x_save_path,
            f"{phase}_predict_flow_y": predict_flow_y_save_path,
            f"{phase}_avg_blend_image_pred": avg_blend_image_pred_save_path,
        }

        if self.use_wandb:
            wandb.log({
                f"{phase}_result": wandb.Image(save_path),
                f"{phase}_flow_x": wandb.Image(flow_x_save_path),
                f"{phase}_flow_y": wandb.Image(flow_y_save_path),
                f"{phase}_predict_flow_x": wandb.Image(predict_flow_x_save_path),
                f"{phase}_predict_flow_y": wandb.Image(predict_flow_y_save_path),
                f"{phase}_avg_blend_image_pred": wandb.Image(avg_blend_image_pred_save_path),
            }, step=self.total_steps)

        return save_path_dict


    
    def upload_test_metrics(self, metrics):
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)
        metircs_str = "TEST: "
        for k in metrics:
            metircs_str += "{}: {:10.4f}, ".format(k, metrics[k])
        loguru_logger.info(metircs_str)

        
        

    def write_dict(self, results):
        pass
        # if self.writer is None:
        #     self.writer = SummaryWriter()

        # for key in results:
        #     self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()

