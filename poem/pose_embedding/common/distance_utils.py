import torch
import numpy as np
import torch.distributions as dists
from pose_embedding.common import data_utils
from scipy.stats import chi2


def compute_l2_distances(lhs_points, rhs_points, squared=False, keepdims=False):
    # Compute squared L2 distances.
    squared_l2_distances = torch.sum(
        (lhs_points - rhs_points) ** 2, dim=-1, keepdim=keepdims)

    # Return squared distances or their square root based on the `squared` parameter.
    return squared_l2_distances if squared else torch.sqrt(squared_l2_distances)


def compute_sigmoid_matching_probabilities(inner_distances, a, b, smoothing=0.1):
    """Computes sigmoid matching probabilities.

    We define sigmoid matching probability as:
      P(x_1, x_2) = (1 - s) * sigmoid(-a * d(x_1, x_2) + b) + s / 2
                  = (1 - s) / (1 + exp(a * d(x_1, x_2) - b)) + s / 2,

    in which d(x_1, x_2) is inner distance between x_1 and x_2, and a and b are
    trainable parameters, with s being the smoothing constant.

    Args:
      inner_distances: A tensor for inner distances. Shape = [...].
      a: A tensor for the sigmoid `a` parameter.
      b: A tensor for the sigmoid `b` parameter.
      smoothing: A float for probability smoothing constant. Ignored if
        non-positive.

    Returns:
      A tensor for matching probabilities. Shape = [...].
    """
    p = torch.sigmoid(-a * inner_distances + b)
    if smoothing > 0.0:
        p = (1.0 - smoothing) * p + smoothing / 2.0
    return p


def compute_all_pair_l2_distances(lhs, rhs, squared=False):
    """Computes all-pair (squared) L2 distances.

    Args:
      lhs: A tensor for LHS point groups. Shape = [..., num_lhs_points, point_dim].
      rhs: A tensor for RHS point groups. Shape = [..., num_rhs_points, point_dim].
      squared: A boolean for whether to use squared L2 distance instead.

    Returns:
      distances: A tensor for (squared) L2 distances. Shape = [..., num_lhs_points, num_rhs_points].
    """
    lhs_squared_norms = torch.sum(lhs**2, dim=-1, keepdim=True)
    rhs_squared_norms = torch.sum(
        rhs**2, dim=-1, keepdim=True).transpose(-1, -2)
    dot_products = torch.matmul(lhs, rhs.transpose(-1, -2))
    distances = -2.0 * dot_products + lhs_squared_norms + rhs_squared_norms

    # if not squared:
    #     distances = torch.sqrt(torch.clamp(distances, min=0.0))
    if not squared:
        distances = torch.sqrt(torch.max(distances, torch.tensor(0.0)))

    return distances


def compute_corresponding_pair_l2_distances(lhs, rhs, squared=False):
    """Computes corresponding-pair (squared) L2 distances.

    Args:
      lhs: A tensor for LHS point groups. Shape = [..., num_points, point_dim].
      rhs: A tensor for RHS point groups. Shape = [..., num_points, point_dim].
      squared: A boolean for whether to use squared L2 distance instead.

    Returns:
      distances: A tensor for (squared) L2 distances. Shape = [..., num_points,
        1].
    """
    return compute_l2_distances(lhs, rhs, squared=squared, keepdims=True)


def compute_gaussian_likelihoods(
        means,
        stddevs,
        samples,
        l2_distance_computer=compute_all_pair_l2_distances,
        min_stddev=0.0,
        max_squared_mahalanobis_distance=0.0,
        smoothing=0.0):
    """Computes sample likelihoods with respect to Gaussian distributions.

    Args:
      means: A tensor for Gaussian means. Shape = [..., 1, sample_dim].
      stddevs: A tensor for Gaussian stddevs. Shape = [..., 1, sample_dim].
      samples: A tensor for samples. Shape = [..., num_samples, sample_dim].
      l2_distance_computer: A function handle for L2 distance computer to use.
      min_stddev: A float for minimum standard deviation to use. Ignored if
        non-positive.
      max_squared_mahalanobis_distance: A float for maximum inner squared
        mahalanobis distance to use. Larger distances will be clipped. Ignored if
        non-positive.
      smoothing: A float for probability smoothing constant. Ignored if
        non-positive.

    Returns:
      A tensor for sample likelihoods. Shape = [..., num_samples].
    """
    if min_stddev > 0.0:
        stddevs = torch.maximum(torch.tensor(min_stddev), stddevs)

    samples *= torch.reciprocal(stddevs)
    means *= torch.reciprocal(stddevs)
    squared_mahalanobis_distances = l2_distance_computer(
        samples, means, squared=True)

    if max_squared_mahalanobis_distance > 0.0:
        squared_mahalanobis_distances = torch.clamp(
            squared_mahalanobis_distances,
            min=0.0,
            max=max_squared_mahalanobis_distance)

    # Use scipy's chi2 cdf function
    df = means.shape[-1]
    p_numpy = 1.0 - chi2.cdf(squared_mahalanobis_distances.cpu().numpy(), df)
    p = torch.tensor(p_numpy, dtype=squared_mahalanobis_distances.dtype,
                     device=squared_mahalanobis_distances.device)

    # chi2 = dists.Chi2(df=means.shape[-1])
    # p = 1.0 - chi2.cdf(squared_mahalanobis_distances)

    if smoothing > 0.0:
        p = (1.0 - smoothing) * p + smoothing / 2.0

    return p


def compute_distance_matrix(start_points,
                            end_points,
                            distance_fn,
                            start_point_masks=None,
                            end_point_masks=None):
    """Computes all-pair distance matrix.

    Computes distance matrix as:
      [d(s_1, e_1),  d(s_1, e_2),  ...,  d(s_1, e_N)]
      [d(s_2, e_1),  d(s_2, e_2),  ...,  d(s_2, e_N)]
      [...,          ...,          ...,  ...        ]
      [d(s_M, e_1),  d(s_2, e_2),  ...,  d(s_2, e_N)]

    Args:
      start_points: A tensor for start points. Shape = [num_start_points, ...,
        point_dim].
      end_points: A tensor for end_points. Shape = [num_end_points, ...,
        point_dim].
      distance_fn: A function handle for computing distance matrix, which takes
        two matrix point tensors and a mask matrix tensor, and returns an
        element-wise distance matrix tensor.
      start_point_masks: A tensor for start point masks. Shape =
        [num_start_points, ...].
      end_point_masks: A tensor for end point masks. Shape = [num_end_points,
        ...].

    Returns:
      A tensor for distance matrix. Shape = [num_start_points, num_end_points,
        ...].
    """
    def expand_and_tile_axis_01(x, target_axis, target_dim):
        """Expands and tiles tensor along target axis 0 or 1."""
        if target_axis not in [0, 1]:
            raise ValueError('Only supports 0 or 1 as target axis: %s.' %
                             str(target_axis))

        # Using `unsqueeze` to expand dimensions in PyTorch
        x = x.unsqueeze(target_axis)

        first_dim_multiples = [1, 1]
        first_dim_multiples[target_axis] = target_dim

        # Directly calling `tile_first_dims` instead of `data_utils.tile_first_dims`
        return data_utils.tile_first_dims(x, first_dim_multiples=first_dim_multiples)

    # def expand_and_tile_axis_01(x, target_axis, target_dim):
    #     """Expands and tiles tensor along target axis 0 or 1."""
    #     if target_axis not in [0, 1]:
    #         raise ValueError('Only supports 0 or 1 as target axis: %s.' %
    #                          str(target_axis))
    #     x = torch.unsqueeze(x, dim=target_axis)
    #     first_dim_multiples = [1, 1]
    #     first_dim_multiples[target_axis] = target_dim
    #     return x.repeat(*first_dim_multiples)

    num_start_points = start_points.shape[0]
    num_end_points = end_points.shape[0]
    start_points = expand_and_tile_axis_01(
        start_points, target_axis=1, target_dim=num_end_points)
    end_points = expand_and_tile_axis_01(
        end_points, target_axis=0, target_dim=num_start_points)

    if start_point_masks is None and end_point_masks is None:
        return distance_fn(start_points, end_points)

    point_masks = None
    if start_point_masks is not None and end_point_masks is not None:
        start_point_masks = expand_and_tile_axis_01(
            start_point_masks, target_axis=1, target_dim=num_end_points)
        end_point_masks = expand_and_tile_axis_01(
            end_point_masks, target_axis=0, target_dim=num_start_points)
        point_masks = start_point_masks * end_point_masks
    elif start_point_masks is not None:
        start_point_masks = expand_and_tile_axis_01(
            start_point_masks, target_axis=1, target_dim=num_end_points)
        point_masks = start_point_masks
    else:  # End_point_masks is not None.
        end_point_masks = expand_and_tile_axis_01(
            end_point_masks, target_axis=0, target_dim=num_start_points)
        point_masks = end_point_masks

    return distance_fn(start_points, end_points, point_masks)


def compute_gaussian_kl_divergence(lhs_means, lhs_stddevs, rhs_means=0.0, rhs_stddevs=1.0):
    """Computes Kullback-Leibler divergence between two multivariate Gaussians.

    Only supports Gaussians with diagonal covariance matrix.

    Args:
      lhs_means: A tensor for LHS Gaussian means. Shape = [..., dim].
      lhs_stddevs: A tensor for LHS Gaussian standard deviations. Shape = [...,
        dim].
      rhs_means: A tensor or a float for LHS Gaussian means. Shape = [..., dim].
      rhs_stddevs: A tensor or a float for LHS Gaussian standard deviations. Shape
        = [..., dim].

    Returns:
      A tensor for KL divergence. Shape = [].
    """
    return 0.5 * torch.sum(
        (lhs_stddevs**2 + (rhs_means - lhs_means)**2)
        / torch.max(rhs_stddevs**2, torch.tensor(1e-12)) - 1.0
        + 2.0 * torch.log(torch.max(rhs_stddevs, torch.tensor(1e-12)))
        - 2.0 * torch.log(torch.max(lhs_stddevs, torch.tensor(1e-12))),
        dim=-1)
