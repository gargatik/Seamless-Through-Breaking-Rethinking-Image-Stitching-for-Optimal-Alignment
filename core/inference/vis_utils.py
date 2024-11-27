import numpy as np
import torch
import matplotlib.pyplot as plt
def plot_quiver(flow, step=20, show_quiver=False): # B,C,H,W
    plt.imshow(flow.norm(dim=1)[0].detach().cpu().numpy())
    plt.colorbar()
    plt.show()
    
    if show_quiver:
        flow = flow[0].detach().cpu().permute(1,2,0).numpy()
        plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], -1, -step), 
               flow[::step, ::step, 0], flow[::step, ::step, 1])
        plt.show()