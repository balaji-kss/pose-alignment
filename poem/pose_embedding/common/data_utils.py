import numpy as np
import torch


def flatten_last_dims(x, num_last_dims):
    """Flattens the last dimensions of a tensor.

    For example:
      x.shape == torch.Size([1, 2, 3, 4])
      flatten_last_dims(x, num_last_dims=2).shape == torch.Size([1, 2, 12])

    Args:
      x: A tensor to flatten.
      num_last_dims: An integer for the number of last dimensions to flatten.

    Returns:
      A flattened tensor.

    Raises:
      ValueError: If `num_last_dims` is greater than the total dimensionality of
        `x`.
    """
    
    shape_list = list(x.shape)
    
    if num_last_dims > len(shape_list):
        raise ValueError(
            'Number of last dimensions must not be greater than the total '
            'dimensionality of input tensor: %d vs. %d.' %
            (num_last_dims, len(shape_list)))

    last_dim = np.prod(shape_list[-num_last_dims:])
    shape_list = shape_list[:-num_last_dims] + [last_dim]

    return x.reshape(shape_list)


def reduce_weighted_mean(tensor, weights=None, axis=None, keepdims=False):
    """Reduces weighted means.

    Args:
      tensor: A tensor to reduce.
      weights: A tensor for weights. Must be non-negative and multiplicable to
        tensor. If None, this function falls back to torch.mean.
      axis: An integer or tuple of integers for axes to reduce along.
      keepdims: A boolean for whether to keep the original dimension for results.

    Returns:
      A reduced tensor.
    """
    if weights is None:
        return torch.mean(tensor, dim=axis, keepdim=keepdims)

    weighted_sum = torch.sum(tensor * weights, dim=axis, keepdim=keepdims)
    sum_weights = torch.sum(weights, dim=axis, keepdim=keepdims)

    return weighted_sum / torch.clamp(sum_weights, min=1e-12)


def tile_first_dims(x, first_dim_multiples):
    """Tiles the first dimensions of a tensor.

    For example:
      x.shape = torch.Size([2, 3, 4]).
      tile_first_dims(x, first_dim_multiples=[2, 3]).shape == torch.Size([4, 9, 4]).

    Args:
      x: A tensor to tile.
      first_dim_multiples: A list of integers for the multiples to tile the first
        dimensions.

    Returns:
      A tiled tensor.

    Raises:
      ValueError: If length of `first_dim_multiples` is greater than the total
        dimensionality of `x`.
    """
    shape_list = list(x.shape)
    num_first_dims = len(first_dim_multiples)
    if num_first_dims > len(shape_list):
        raise ValueError(
            'Number of first dimensions must not be greater than the total '
            'dimensionality of input tensor: %d vs. %d.' %
            (num_first_dims, len(shape_list)))
    num_other_dims = len(shape_list) - num_first_dims
    multiples = list(first_dim_multiples) + [1] * num_other_dims

    # PyTorch doesn't have a direct function similar to tf.tile for higher dimensions, so we'll use repeat
    return x.repeat(*multiples)

def tile_last_dims(x, last_dim_multiples):
    """Tiles the last dimensions of a tensor.

    For example:
      x.shape == torch.Size([2, 3, 4])
      tile_last_dims(x, last_dim_multiples=[2, 3]).shape == torch.Size([2, 6, 12])

    Args:
      x: A tensor to tile.
      last_dim_multiples: A list of integers for the multiples to tile the last
        dimensions.

    Returns:
      A tiled tensor.

    Raises:
      ValueError: If length of `last_dim_multiples` is greater than the total
        dimensionality of `x`.
    """    
    shape_list = list(x.shape)
    num_last_dims = len(last_dim_multiples)
    if num_last_dims > len(shape_list):
        raise ValueError(
            'Number of last dimensions must not be greater than the total '
            'dimensionality of input tensor: %d vs. %d.' %
            (num_last_dims, len(shape_list)))

    multiples = [1] * (len(shape_list) - num_last_dims) + list(last_dim_multiples)

    return x.repeat(multiples)


def reshape_by_last_dims(x, last_dim_shape):
    """
    Reshapes a tensor by the last dimensions.
    """
    # new_shape = list(x.shape[:-len(last_dim_shape)]) + last_dim_shape
    new_shape = list(x.shape[:-len(last_dim_shape)]) + list(last_dim_shape)
    return x.reshape(new_shape)

def recursively_expand_dims(x, axes):
    """
    Recursively applies a sequence of axis expansion on a tensor.
    """
    for axis in axes:
        x = torch.unsqueeze(x, dim=axis)
    return x


def flatten_first_dims(x, num_last_dims_to_keep):
    """Flattens the first dimensions of a tensor.

    For example:
      x.shape == torch.Size([1, 2, 3, 4, 5])
      flatten_first_dims(x, num_last_dims_to_keep=2).shape == torch.Size([6, 4, 5])

    Returns:
      A flattened tensor.

    Args:
      x: A tensor to flatten.
      num_last_dims_to_keep: An integer for the number of last dimensions to keep.
        The rest of the preceding dimensions will be flattened.

    Raises:
      ValueError: If `num_last_dims_to_keep` is greater than the total
        dimensionality of `x`.
    """
    if num_last_dims_to_keep > len(x.shape):
        raise ValueError(
            'Number of last dimensions must not be greater than the total '
            'dimensionality of input tensor: %d vs. %d.' %
            (num_last_dims_to_keep, len(x.shape)))

    new_shape = [-1] + list(x.shape[-num_last_dims_to_keep:])
    return torch.reshape(x, new_shape)

def unflatten_first_dim(x, shape_to_unflatten):
    """Unflattens the first dimension of a tensor.

    For example:
      x.shape == torch.Size([6, 2])
      unflatten_first_dim(x, [2, 3]).shape == torch.Size([2, 3, 2])

    Args:
      x: A tensor to unflatten.
      shape_to_unflatten: A list of integers to reshape the first dimension of `x`
        into.

    Returns:
      An unflattened tensor.
    """
    new_shape = list(shape_to_unflatten) + list(x.shape[1:])
    return torch.reshape(x, new_shape)

def get_shape_by_first_dims(x, num_last_dims):
    """Gets tensor shape by the first dimensions.

    For example:
      x.shape == torch.Size([1, 2, 3, 4, 5])
      get_shape_by_first_dims(x, num_last_dims=2) == [1, 2, 3]

    Args:
      x: A tensor to get shape of.
      num_last_dims: An integer for the number of last dimensions not to get shape
        of.

    Returns:
      A list for tensor shape.
    """
    return list(x.shape[:-num_last_dims])


def sample_gaussians(means, stddevs, num_samples, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    means = tile_last_dims(means.unsqueeze(-2), last_dim_multiples=[num_samples, 1])
    stddevs = tile_last_dims(stddevs.unsqueeze(-2), last_dim_multiples=[num_samples, 1])
    epsilons = torch.normal(0, 1, size=means.shape, device=means.device)
    
    return epsilons * stddevs + means

def mix_batch(lhs_batches, rhs_batches, axis, assignment=None,
              keep_lhs_prob=0.5, seed=None):
    """Mixes batches.

    A pair of tensors from the same location in each list are assumed to have the
    same shape.

    Example:
      # Shape = [4, 3, 2, 1].
      lhs_batches[0] = [[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                        [[[2.0], [2.1]], [[2.2], [2.3]], [[2.4], [2.5]]],
                        [[[3.0], [3.1]], [[3.2], [3.3]], [[3.4], [3.5]]],
                        [[[4.0], [4.1]], [[4.2], [4.3]], [[4.4], [4.5]]]]
      rhs_batches[0] = [[[[11.0], [11.1]], [[11.2], [11.3]], [[11.4], [11.5]]],
                        [[[12.0], [12.1]], [[12.2], [12.3]], [[12.4], [12.5]]],
                        [[[13.0], [13.1]], [[13.2], [13.3]], [[13.4], [13.5]]],
                        [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]]

      # Shape = [4, 3, 2, 2, 1].
      lhs_batches[1] = [[[[[1.0], [10.0]], [[1.1], [10.1]]],
                         [[[1.2], [10.2]], [[1.3], [10.3]]],
                         [[[1.4], [10.4]], [[1.5], [10.5]]]],
                        [[[[2.0], [20.0]], [[2.1], [20.1]]],
                         [[[2.2], [20.2]], [[2.3], [20.3]]],
                         [[[2.4], [20.4]], [[2.5], [20.5]]]],
                        [[[[3.0], [30.0]], [[3.1], [30.1]]],
                         [[[3.2], [30.2]], [[3.3], [30.3]]],
                         [[[3.4], [30.4]], [[3.5], [30.5]]]],
                        [[[[4.0], [40.0]], [[4.1], [40.1]]],
                         [[[4.2], [40.2]], [[4.3], [40.3]]],
                         [[[4.4], [40.4]], [[4.5], [40.5]]]]]
      rhs_batches[1] = [[[[[11.0], [110.0]], [[11.1], [110.1]]],
                         [[[11.2], [110.2]], [[11.3], [110.3]]],
                         [[[11.4], [110.4]], [[11.5], [110.5]]]],
                        [[[[12.0], [120.0]], [[12.1], [120.1]]],
                         [[[12.2], [120.2]], [[12.3], [120.3]]],
                         [[[12.4], [120.4]], [[12.5], [120.5]]]],
                        [[[[13.0], [130.0]], [[13.1], [130.1]]],
                         [[[13.2], [130.2]], [[13.3], [130.3]]],
                         [[[13.4], [130.4]], [[13.5], [130.5]]]],
                        [[[[14.0], [140.0]], [[14.1], [140.1]]],
                         [[[14.2], [140.2]], [[14.3], [140.3]]],
                         [[[14.4], [140.4]], [[14.5], [140.5]]]]]

      # Shape = [4, 1, 2].
      assignment = [[[True, True]], [[True, False]],
                    [[False, True]], [[False, False]]]
      axis = 2
      -->
      # Shape = [4, 3, 2, 1].
      mixed_batches[0] = [[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                          [[[2.0], [12.1]], [[2.2], [12.3]], [[2.4], [12.5]]],
                          [[[13.0], [3.1]], [[13.2], [3.3]], [[13.4], [3.5]]],
                          [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]]

      # Shape = [4, 3, 2, 2, 1].
      mixed_batches[1] = [[[[[1.0], [10.0]], [[1.1], [10.1]]],
                           [[[1.2], [10.2]], [[1.3], [10.3]]],
                           [[[1.4], [10.4]], [[1.5], [10.5]]]],
                          [[[[2.0], [20.0]], [[12.1], [120.1]]],
                           [[[2.2], [20.2]], [[12.3], [120.3]]],
                           [[[2.4], [20.4]], [[12.5], [120.5]]]],
                          [[[[13.0], [130.0]], [[3.1], [30.1]]],
                           [[[13.2], [130.2]], [[3.3], [30.3]]],
                           [[[13.4], [130.4]], [[3.5], [30.5]]]],
                          [[[[14.0], [140.0]], [[14.1], [140.1]]],
                           [[[14.2], [140.2]], [[14.3], [140.3]]],
                           [[[14.4], [140.4]], [[14.5], [140.5]]]]]

    Args:
      lhs_batches: A list of tensors for LHS batches. Each tensor shape =
        [batch_size, ..., num_instances, ...].
      rhs_batches: A list of tensors for RHS batches. Each tensor shape =
        [batch_size, ..., num_instances, ...].
      axis: An integer for the mixing axis (the `num_instances` dimension).
      assignment: A tensor for assignment indicator matrix. Shape = [batch_size,
        ..., num_instances]. A True/False value indicates element from the LHS/RHS
        tensor will be kept at the corresponding location. For the "idle
        dimensions" between the batch dimension (0) and the mixing axis dimension,
        size 1 can be used to take advantage of broadcasting. If None, A uniformly
        random assignment matrix will be created.
      keep_lhs_prob: A float indicates the probability to randomly keep
        lhs_batches along axis. This is only useful when `assignment` is None.
      seed: An integer for random seed.

    Returns:
      mixed_batches: A list of tensors for mixed batches. Each tensor shape =
        [batch_size, ..., num_instances, ...].

    Raises:
      ValueError: If `lhs_batches` and `rhs_batches` have different sizes.
      ValueError: If `lhs_batches` or `rhs_batches` is empty.
      ValueError: If `axis` is out of range or incompatible with `assignment`.

    """
    if len(lhs_batches) != len(rhs_batches):
        raise ValueError(
            '`Lhs_batches` and `rhs_batches` size disagree: %d vs. %d.' %
            (len(lhs_batches), len(rhs_batches)))
    if not lhs_batches:
        raise ValueError('Tensor lists are empty.')

    if seed is not None:
        torch.manual_seed(seed)

    def get_random_assignment(batch_shape):
        """Gets batch-compatible assignment."""
        assignment_shape = [1] * (axis + 1)
        assignment_shape[0] = batch_shape[0]
        assignment_shape[axis] = batch_shape[axis]
        assignment = torch.rand(assignment_shape)
        assignment = assignment >= keep_lhs_prob
        # assignment = tf.random.uniform(
        #     assignment_shape, minval=0.0, maxval=1.0, seed=seed)
        # assignment = tf.math.greater_equal(assignment, keep_lhs_prob)
        return assignment

    if assignment is None:
        first_batch_shape = list(lhs_batches[0].shape)
        assignment = get_random_assignment(first_batch_shape)
    else:
        assignment_rank = len(assignment.shape)
        if assignment_rank != axis + 1:
            raise ValueError('`Assignment` and `axis` are incompatible: %d vs. %d.' %
                             (assignment_rank, axis))

    mixed_batches = []
    for i, batch_pair in enumerate(zip(lhs_batches, rhs_batches)):
        lhs_batch, rhs_batch = batch_pair
        batch_rank = len(lhs_batch.shape)
        if axis < 0 or batch_rank <= axis:
            raise ValueError('Axis out of range for the %d-th tensor: %d.' %
                             (i, axis))
        if batch_rank != len(rhs_batch.shape):
            raise ValueError(
                'The %d-th LHS/RHS tensor have different ranks: %d vs. %d.' %
                (i, batch_rank, len(rhs_batch.shape)))

        assignment_rank = axis + 1
        if len(lhs_batch.shape) > assignment_rank:
            batch_assignment = recursively_expand_dims(
                assignment, axes=[-1] * (batch_rank - assignment_rank))
        else:
            batch_assignment = assignment
        
        mixed_batches.append(torch.where(batch_assignment, lhs_batch, rhs_batch))
        # mixed_batches.append(tf.where(batch_assignment, lhs_batch, rhs_batch))

    return mixed_batches
