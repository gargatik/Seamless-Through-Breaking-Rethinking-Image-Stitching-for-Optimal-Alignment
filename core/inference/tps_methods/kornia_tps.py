from kornia.geometry.transform import warp_image_tps, get_tps_transform, warp_points_tps
from typing import Tuple

import torch
import torch.nn as nn

from kornia.utils import create_meshgrid
# from kornia.utils.helpers import _torch_solve_cast
from kornia.core import Tensor
def _torch_solve_cast(A: Tensor, B: Tensor) -> Tensor:
    """Helper function to make torch.solve work with other than fp32/64.

    The function torch.solve is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.svd, and cast back to the input dtype.
    """
    dtype: torch.dtype = A.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    # out = torch.linalg.solve(A.to(dtype), B.to(dtype))
    # use psudo inverse
    out = torch.pinverse(A.to(dtype)).matmul(B.to(dtype))

    return out.to(A.dtype)

def _pair_square_euclidean(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    r"""Compute the pairwise squared euclidean distance matrices :math:`(B, N, M)` between two tensors with shapes
    (B, N, C) and (B, M, C)."""
    # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
    t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
    t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
    t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
    square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
    square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
    return square_dist


def _kernel_distance(squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Compute the TPS kernel distance function: :math:`r^2 log(r)`, where `r` is the euclidean distance.

    Since :math:`\log(r) = 1/2 \log(r^2)`, this function takes the squared distance matrix and calculates
    :math:`0.5 r^2 log(r^2)`.
    """
    # r^2 * log(r) = 1/2 * r^2 * log(r^2)
    return 0.5 * squared_distances * squared_distances.add(eps).log()

def custom_get_tps_transform(points_src: torch.Tensor, points_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the TPS transform parameters that warp source points to target points.

    The input to this function is a tensor of :math:`(x, y)` source points :math:`(B, N, 2)` and a corresponding
    tensor of target :math:`(x, y)` points :math:`(B, N, 2)`.

    Args:
        points_src: batch of source points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.
        points_dst: batch of target points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.

    Returns:
        :math:`(B, N, 2)` tensor of kernel weights and :math:`(B, 3, 2)`
            tensor of affine weights. The last dimension contains the x-transform and y-transform weights
            as separate columns.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

    .. note::
        This function is often used in conjunction with :func:`warp_points_tps`, :func:`warp_image_tps`.
    """
    if not isinstance(points_src, torch.Tensor):
        raise TypeError(f"Input points_src is not torch.Tensor. Got {type(points_src)}")

    if not isinstance(points_dst, torch.Tensor):
        raise TypeError(f"Input points_dst is not torch.Tensor. Got {type(points_dst)}")

    if not len(points_src.shape) == 3:
        raise ValueError(f"Invalid shape for points_src, expected BxNx2. Got {points_src.shape}")

    if not len(points_dst.shape) == 3:
        raise ValueError(f"Invalid shape for points_dst, expected BxNx2. Got {points_dst.shape}")

    device, dtype = points_src.device, points_src.dtype
    batch_size, num_points = points_src.shape[:2]

    # set up and solve linear system
    # [K   P] [w] = [dst]
    # [P^T 0] [a]   [ 0 ]
    pair_distance: torch.Tensor = _pair_square_euclidean(points_src, points_dst)
    k_matrix: torch.Tensor = _kernel_distance(pair_distance)

    zero_mat: torch.Tensor = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    one_mat: torch.Tensor = torch.ones(batch_size, num_points, 1, device=device, dtype=dtype)
    dest_with_zeros: torch.Tensor = torch.cat((points_dst, zero_mat[:, :, :2]), 1)
    p_matrix: torch.Tensor = torch.cat((one_mat, points_src), -1)
    p_matrix_t: torch.Tensor = torch.cat((p_matrix, zero_mat), 1).transpose(1, 2)
    l_matrix: torch.Tensor = torch.cat((k_matrix, p_matrix), -1)
    l_matrix = torch.cat((l_matrix, p_matrix_t), 1)

    weights = _torch_solve_cast(l_matrix, dest_with_zeros)
    kernel_weights: torch.Tensor = weights[:, :-3]
    affine_weights: torch.Tensor = weights[:, -3:]

    return (kernel_weights, affine_weights)
from kornia.utils import create_meshgrid
def warp_image_tps(
    image: torch.Tensor,
    kernel_centers: torch.Tensor,
    kernel_weights: torch.Tensor,
    affine_weights: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Warp an image tensor according to the thin plate spline transform defined by kernel centers, kernel weights,
    and affine weights.

    .. image:: _static/img/warp_image_tps.png

    The transform is applied to each pixel coordinate in the output image to obtain a point in the input
    image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
    output space to input space.

    The input `image` is a :math:`(B, C, H, W)` tensor. The kernel centers, kernel weight and affine weights
    are the same as in `warp_points_tps`.

    Args:
        image: input image tensor :math:`(B, C, H, W)`.
        kernel_centers: kernel center points :math:`(B, K, 2)`.
        kernel_weights: tensor of kernl weights :math:`(B, K, 2)`.
        affine_weights: tensor of affine weights :math:`(B, 3, 2)`.
        align_corners: interpolation flag used by `grid_sample`.

    Returns:
        warped image tensor :math:`(B, C, H, W)`.

    Example:
        >>> points_src = torch.rand(1, 5, 2)
        >>> points_dst = torch.rand(1, 5, 2)
        >>> image = torch.rand(1, 3, 32, 32)
        >>> # note that we are getting the reverse transform: dst -> src
        >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
        >>> warped_image = warp_image_tps(image, points_src, kernel_weights, affine_weights)

    .. note::
        This function is often used in conjunction with :func:`get_tps_transform`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

    if not isinstance(kernel_centers, torch.Tensor):
        raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

    if not isinstance(kernel_weights, torch.Tensor):
        raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

    if not isinstance(affine_weights, torch.Tensor):
        raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

    if not len(kernel_centers.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

    if not len(kernel_weights.shape) == 3:
        raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

    if not len(affine_weights.shape) == 3:
        raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

    batch_size, _, h, w = image.shape
    coords: torch.Tensor = create_meshgrid(h, w, device=image.device, dtype=image.dtype)
    coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
    warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
    warped = warped.view(-1, h, w, 2)
    warped_image: torch.Tensor = nn.functional.grid_sample(image, warped, align_corners=align_corners)

    return warped_image