import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def advanced_uniform_sample_border_points(image, step, pad_num, is_plot=False):
    """
    Advanced uniform sampling of border points.

    This function performs advanced uniform sampling on the border points of an image. It first uniformly samples
    points along the border and then selects the points with the highest gradient in each sampled range.

    Args:
        image (torch.Tensor): The input image to calculate the gradient.
        step (int): The sampling step size.
        pad_num (int): The number of padding pixels.
        is_plot (bool, optional): Whether to plot the sampled points. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the coordinates of the selected high gradient points.
    """

    # advanced uniform sample
    # uniform sample first and select high gradient points
    # from [0, H] sample $step points,  we can get the range of the order points
    # e.g [ 0, 10] sample 5 points, we can get [0,2,4,6,8,10], the ranges are [0,2], [2,4], [4,6], [6,8], [8,10]
    # and we can get the gradient of every range, and select the max gradient point in every range
    # and we can get the high gradient points
    # e.g. from [0,2], [2,4], [4,6], [6,8], [8,10], we can get the max gradient point in every range, e.g [0], [3], [4], [8], [9]
    # and we can get the high gradient points [0,3,4,8,9]
    # let's extend this method to 2d
    # image: to calculate gradient
    # step: sample step
    # pad_num: pad num
    # num_points: sample num points
    _, _, H, W = image.shape
    border_points = []
    range_points = []
    # top
    i_old = 0
    for i in range(0+pad_num,W-pad_num,step):
        border_points.append([i, 0 + pad_num])
        if i_old !=0:
            range_points.append([ (i_old, 0 + pad_num), (i, 0 + pad_num)] )
        i_old = i
    # bottom
    i_old = 0
    for i in range(0+pad_num,W-pad_num,step):
        border_points.append([i, H-1 - pad_num])
        if i_old !=0:
            range_points.append([ (i_old, H-1 - pad_num), (i, H-1 - pad_num)] )
        i_old = i
    # left
    i_old = 0
    for i in range(0+pad_num,H-pad_num,step):
        border_points.append([0 + pad_num, i])
        if i_old !=0:
            range_points.append([ (0 + pad_num, i_old), (0 + pad_num, i)] )
        i_old = i
    # right
    i_old = 0
    for i in range(0+pad_num,H-pad_num,step):
        border_points.append([W-1 - pad_num, i])
        if i_old !=0:
            range_points.append([ (W-1 - pad_num, i_old), (W-1 - pad_num, i)] )
        i_old = i
    # calculate gradient, keep the H,W is the same as image, and the result is only one channel
    # image shape: (B, C, H, W)
    # image_grad = torch.abs(F.conv2d(image, torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().to(image.device).view(1,1,3,3), padding=1, groups=1))
    # image shape: (B, C, H, W)
    # horizontal gradient
    kernel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().to(image.device)
    # Repeat the kernel for each input channel
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)
    image_grad_horizontal = torch.abs(F.conv2d(image, kernel, padding=1, groups=image.shape[1]))
    image_grad_horizontal = image_grad_horizontal.mean(dim=1, keepdim=True)

    # vertical gradient
    kernel = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().to(image.device)
    # Repeat the kernel for each input channel
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)
    image_grad_vertical = torch.abs(F.conv2d(image, kernel, padding=1, groups=image.shape[1]))
    image_grad_vertical = image_grad_vertical.mean(dim=1, keepdim=True)

    # make image_grad to one channel
    image_grad = image_grad_horizontal.abs() + image_grad_vertical.abs()
    

    # get the max gradient point in every range
    max_grad_points = []
    for range_point in range_points:
        x1, y1, x2, y2 = range_point[0][0], range_point[0][1], range_point[1][0], range_point[1][1]
        # x for W, y for H
        mask_grad = torch.zeros_like(image_grad)
        # mask the range
        # mask_grad[:,:,y1:y2+16,x1:x2+16] = 1
        mask_grad[:,:,y1-2:y2+2,x1-2:x2+2] = 1
        range_grad = image_grad * mask_grad + ( (-1 * torch.ones_like(image_grad)) * (1 - mask_grad))
        
        if range_grad.numel() > 0:
            max_grad_point_index = torch.argmax(range_grad)
            all_zero_only_max_is_one = torch.zeros_like(image_grad)
            all_zero_only_max_is_one.view(-1)[max_grad_point_index] = 1
           
            max_grad_point = torch.nonzero(all_zero_only_max_is_one)
            
            max_grad_point = max_grad_point[0][2:] # B,C,H,W --> H,W
            y, x  = max_grad_point[0], max_grad_point[1]
            max_grad_point = [x, y]
            max_grad_points.append(max_grad_point)
        else:
            print("Warning: range_grad is empty.")
    border_points = torch.tensor(border_points)
    border_points = torch.unique(border_points, dim=0)
    max_grad_points = torch.tensor(max_grad_points)
    max_grad_points = torch.unique(max_grad_points, dim=0)
    
    if is_plot:
        print("max_grad_points",max_grad_points.shape, "border_points",border_points.shape)
        print("max_grad_points",max_grad_points)
        plt.imshow(image_grad[0,0].detach().cpu().numpy(),cmap="gray")
        plt.scatter(border_points[:,0], border_points[:,1], s=1, c="red")
        plt.show()
        plt.imshow(image_grad[0,0].detach().cpu().numpy(),cmap="gray")
        plt.scatter(max_grad_points[:,0], max_grad_points[:,1], s=1, c="yellow")
        plt.show()
    
    
    return max_grad_points
