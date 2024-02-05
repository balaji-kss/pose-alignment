import numpy as np
import torch
import functools
import torch.nn.functional as F

from pose_embedding.common import data_utils, distance_utils, keypoint_utils, constants

def compute_lower_percentile_means(x, axis, q=50):
    """Computes means of elements less or equal to q percentile along axes.

    Args:
      x: A tensor for input values. Shape = [..., dim_1, ..., dim_len(axis)].
      axis: An integer or a list of integers for percentile reduction axis.
      q: A scalar for percentile.

    Returns:
      A tensor for means of elements less or equal to the q percentiles. Shape =
        [...].
    """
    epsilon = 1e-10  # to prevent division by zero
    if not isinstance(axis, (list, tuple)):
        axis = [axis]

    percentiles = x.clone()
    for ax in axis:
        percentiles = torch.quantile(
            percentiles, q / 100.0, dim=ax, keepdim=True)

    weights = (x <= percentiles).float()

    for ax in reversed(axis):
        weighted_sum = torch.sum(x * weights, dim=ax, keepdim=True)
        sum_weights = torch.sum(weights, dim=ax, keepdim=True)

        x = weighted_sum / (sum_weights + epsilon)
        # reduce weights the same way as x
        weights = torch.sum(weights, dim=ax, keepdim=True)

    return x.squeeze()

def get_sigmoid_parameters(raw_a,
                           raw_b,
                           a_range=(None, None),
                           b_range=(None, None)):
    """Gets sigmoid parameter variables in PyTorch."""

    def maybe_clamp(x, x_range, ignored_if_non_positive):
        """Clamps `x` to `x_range`."""
        x_min, x_max = x_range
        if x_min is not None and x_max is not None and x_min > x_max:
            raise ValueError('Invalid range: %s.' % str(x_range))
        if x_min is not None and (not ignored_if_non_positive or x_min > 0.0):
            x = torch.max(torch.full_like(x, x_min), x)
        if x_max is not None and (not ignored_if_non_positive or x_max > 0.0):
            x = torch.min(torch.full_like(x, x_max), x)
        return x

    a = F.elu(raw_a) + 1.0
    a = maybe_clamp(a, a_range, ignored_if_non_positive=True)
    raw_b = maybe_clamp(raw_b, b_range, ignored_if_non_positive=False)

    return a, raw_b

def get_sigmoid_parameters_0(raw_a_initial_value=0.0,
                           b_initial_value=0.0,
                           a_range=(None, None),
                           b_range=(None, None)):
    """Gets sigmoid parameter variables in PyTorch."""

    def maybe_clamp(x, x_range, ignored_if_non_positive):
        """Clamps `x` to `x_range`."""
        x_min, x_max = x_range
        if x_min is not None and x_max is not None and x_min > x_max:
            raise ValueError('Invalid range: %s.' % str(x_range))
        if x_min is not None and (not ignored_if_non_positive or x_min > 0.0):
            x = torch.max(torch.full_like(x, x_min), x)
        if x_max is not None and (not ignored_if_non_positive or x_max > 0.0):
            x = torch.min(torch.full_like(x, x_max), x)
        return x

    raw_a = torch.nn.Parameter(torch.tensor(raw_a_initial_value))
    a = F.elu(raw_a) + 1.0
    a = maybe_clamp(a, a_range, ignored_if_non_positive=True)

    b = torch.nn.Parameter(torch.tensor(b_initial_value))
    b = maybe_clamp(b, b_range, ignored_if_non_positive=False)

    return raw_a, a, b


def create_sample_distance_fn(
        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
        # distance_kernel=constants.DISTANCE_KERNEL_SQUARED_L2,
        # pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,        
        distance_kernel=constants.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
        pairwise_reduction=constants.DISTANCE_REDUCTION_NEG_LOG_MEAN,
        componentwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
        **distance_kernel_kwargs):
    """Creates sample distance function.

    Args:
      pair_type: An enum string (see `constants`) for type of pairs to use.
      distance_kernel: An enum string (see `constants`) or a function handle for
        point distance kernel to use.
      pairwise_reduction: An enum string (see `constants`) or a function handle for
        pairwise distance reducer to use. If not a supported enum string, uses it
        directly as a function handle.
      componentwise_reduction: An enum string (see `constants`) or a function handle
        for component-wise distance reducer to use. If not a supported enum
        string, uses it directly as a function handle.
      **distance_kernel_kwargs: A dictionary for additional arguments to be passed
        to the distance kernel. The keys are in the format
        `${distance_kernel_name}_${argument_name}`.

    Returns:
      A function handle for computing sample group distances that takes two
        tensors of shape [..., num_components, num_embeddings, embedding_dim] as
        input.
    """

    def get_distance_matrix_fn():
        """Selects point distance matrix function."""
        if pair_type == constants.DISTANCE_PAIR_TYPE_ALL_PAIRS:
            l2_distance_computer = distance_utils.compute_all_pair_l2_distances
        elif pair_type == constants.DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS:
            l2_distance_computer = (
                distance_utils.compute_corresponding_pair_l2_distances)

        if distance_kernel == constants.DISTANCE_KERNEL_SQUARED_L2:
            return functools.partial(l2_distance_computer, squared=True)

        if distance_kernel == constants.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB:

            def compute_l2_sigmoid_matching_distances(lhs, rhs):
                """Computes L2 sigmoid matching probability distances."""
                inner_distances = l2_distance_computer(lhs, rhs, squared=False)
                return distance_utils.compute_sigmoid_matching_probabilities(
                    inner_distances,
                    a=distance_kernel_kwargs.get(distance_kernel + '_a', None),
                    b=distance_kernel_kwargs.get(distance_kernel + '_b', None))

            return compute_l2_sigmoid_matching_distances

        if (distance_kernel ==
                constants.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB):

            def compute_squared_l2_sigmoid_matching_distances(lhs, rhs):
                """Computes squared L2 sigmoid matching probability distances."""
                inner_distances = l2_distance_computer(lhs, rhs, squared=True)
                return distance_utils.compute_sigmoid_matching_probabilities(
                    inner_distances,
                    a=distance_kernel_kwargs.get(distance_kernel + '_a', None),
                    b=distance_kernel_kwargs.get(distance_kernel + '_b', None))

            return compute_squared_l2_sigmoid_matching_distances

        # if distance_kernel == constants.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD:
        #     ###### Currently, we do not use this, so just comment it out
        #     def compute_gaussian_likelihoods(lhs, rhs):
        #         """Computes sample likelihoods."""
        #         num_lhs_samples = lhs.shape[-2] - 2
        #         num_rhs_samples = rhs.shape[-2] - 2
        #         lhs_means, lhs_stddevs, lhs_samples = torch.split(lhs, [1, 1, num_lhs_samples], dim=-2)
        #         rhs_means, rhs_stddevs, rhs_samples = torch.split(rhs, [1, 1, num_rhs_samples], dim=-2)

        #         # Extract arguments from distance_kernel_kwargs using PyTorch's dict get() method
        #         min_stddev = distance_kernel_kwargs.get(distance_kernel + '_min_stddev', None)
        #         max_squared_mahalanobis_distance = distance_kernel_kwargs.get(distance_kernel + '_max_squared_mahalanobis_distance', None)
        #         smoothing = distance_kernel_kwargs.get(distance_kernel + '_smoothing', None)

        #         # Call the provided compute_gaussian_likelihoods from distance_utils
        #         rhs_likelihoods = distance_utils.compute_gaussian_likelihoods(
        #             lhs_means,
        #             lhs_stddevs,
        #             rhs_samples,
        #             min_stddev=min_stddev,
        #             max_squared_mahalanobis_distance=max_squared_mahalanobis_distance,
        #             smoothing=smoothing)

        #         lhs_likelihoods = distance_utils.compute_gaussian_likelihoods(
        #             rhs_means,
        #             rhs_stddevs,
        #             lhs_samples,
        #             l2_distance_computer=l2_distance_computer,
        #             min_stddev=min_stddev,
        #             max_squared_mahalanobis_distance=max_squared_mahalanobis_distance,
        #             smoothing=smoothing)

        #         return (rhs_likelihoods + lhs_likelihoods) / 2.0
        #     return compute_gaussian_likelihoods

        raise ValueError('Unsupported distance kernel: `%s`.' %
                         str(distance_kernel))

    def get_pairwise_distance_reduction_fn():
        """Selects pairwise distance reduction function."""
        if pairwise_reduction == constants.DISTANCE_REDUCTION_MEAN:
            return functools.partial(torch.mean, dim=[-2, -1])

        if pairwise_reduction == constants.DISTANCE_REDUCTION_LOWER_HALF_MEAN:
            return functools.partial(compute_lower_percentile_means, axis=[-2, -1], q=50)

        if pairwise_reduction == constants.DISTANCE_REDUCTION_NEG_LOG_MEAN:
            return lambda x: -torch.log(torch.mean(x, dim=[-2, -1]))

        if pairwise_reduction == constants.DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN:
            def compute_lower_half_negative_log_mean(x):
                return -torch.log(compute_lower_percentile_means(x, axis=[-2, -1], q=50))
            return compute_lower_half_negative_log_mean

        if pairwise_reduction == constants.DISTANCE_REDUCTION_ONE_MINUS_MEAN:
            return lambda x: 1.0 - torch.mean(x, dim=[-2, -1])

        return pairwise_reduction

    def get_componentwise_distance_reduction_fn():
        """Selects component-wise distance reduction function."""
        if componentwise_reduction == constants.DISTANCE_REDUCTION_MEAN:
            return functools.partial(torch.mean, dim=[-1])

        return componentwise_reduction

    def sample_distance_fn(lhs, rhs):
        """Computes sample distances."""
        distances = get_distance_matrix_fn()(lhs, rhs)
        distances = get_pairwise_distance_reduction_fn()(distances)
        distances = get_componentwise_distance_reduction_fn()(distances)
        return distances
    
    # def sample_distance_fn_1(lhs, rhs):
    #     distances = get_distance_matrix_fn()(lhs, rhs)
    #     return distances 

    return sample_distance_fn


def compute_negative_indicator_matrix(anchor_points,
                                      match_points,
                                      distance_fn,
                                      min_negative_distance,
                                      anchor_point_masks=None,
                                      match_point_masks=None):
    """Computes all-pair negative match indicator matrix.

    Args:
      anchor_points: A tensor for anchor points. Shape = [num_anchors, ...,
        point_dim].
      match_points: A tensor for match points. Shape = [num_matches, ...,
        point_dim].
      distance_fn: A function handle for computing distance matrix.
      min_negative_distance: A float for the minimum negative distance threshold.
      anchor_point_masks: A tensor for anchor point masks. Shape = [num_anchors,
        ...]. Ignored if None.
      match_point_masks: A tensor for match point masks. Shape = [num_matches,
        ...]. Ignored if None.

    Returns:
      A boolean tensor for negative indicator matrix. Shape = [num_anchors,
        num_matches].
    """
    distance_matrix = distance_utils.compute_distance_matrix(
        anchor_points,
        match_points,
        distance_fn=distance_fn,
        start_point_masks=anchor_point_masks,
        end_point_masks=match_point_masks)
    return distance_matrix >= min_negative_distance


def compute_hard_negative_distances(anchor_match_distance_matrix,
                                    negative_indicator_matrix,
                                    use_semi_hard=False,
                                    anchor_positive_mining_distances=None,
                                    anchor_match_mining_distance_matrix=None):
    """Computes (semi-)hard negative distances in PyTorch.

    Args:
      anchor_match_distance_matrix: A tensor for anchor/match distance matrix.
        Shape = [num_anchors, num_matches].
      negative_indicator_matrix: A tensor for anchor/match negative indicator
        matrix. Shape = [num_anchors, num_matches].
      use_semi_hard: A boolean for whether to compute semi-hard negative distances
        instead of hard negative distances.
      anchor_positive_mining_distances: A tensor for positive distances of each
        anchor for (semi-)hard negative mining. Only used if `use_semi_hard` is
        True. Shape = [num_anchors].
      anchor_match_mining_distance_matrix: A tensor for an alternative
        anchor/match distance matrix to use for (semi-)hard negative mining. Use
        None to ignore and use `anchor_match_distance_matrix` instead. If
        specified, must be of the same shape as `anchor_match_distance_matrix`.

    Returns:
      hard_negative_distances: A tensor for (semi-)hard negative distances. Shape
        = [num_amchors]. If an anchor has no (semi-)hard negative match, its
        negative distance will be assigned as the maximum value of
        anchor_match_distance_matrix.dtype.
      hard_negative_mining_distances: A tensor for (semi-)hard negative mining
        distances. Shape = [num_amchors]. If an anchor has no (semi-)hard negative
        match, its negative distance will be assigned as the maximum value of
        anchor_match_distance_matrix.dtype.

    Raises:
        ValueError: If `use_semi_hard` is True, but `anchor_positive_mining_distances` is not specified.
    """

    indicators = negative_indicator_matrix.clone()

    if anchor_match_mining_distance_matrix is None:
        anchor_match_mining_distance_matrix = anchor_match_distance_matrix.clone()

    if use_semi_hard:
        if anchor_positive_mining_distances is None:
            raise ValueError(
                'Positive match embeddings must be specified to compute semi-hard distances.')
        anchor_positive_mining_distances = anchor_positive_mining_distances.unsqueeze(
            -1)
        indicators &= (anchor_match_mining_distance_matrix >
                       anchor_positive_mining_distances)

    def find_hard_distances(distance_matrix, indicator_matrix):
        distance_matrix = torch.where(
            indicator_matrix,
            distance_matrix,
            torch.full_like(distance_matrix, float('inf'))
        )
        hard_distances, _ = torch.min(distance_matrix, dim=-1)
        return hard_distances

    hard_negative_mining_distances = find_hard_distances(
        anchor_match_mining_distance_matrix, indicators)

    indicators &= (anchor_match_mining_distance_matrix ==
                   hard_negative_mining_distances.unsqueeze(-1))

    hard_negative_distances = find_hard_distances(
        anchor_match_distance_matrix, indicators)

    return hard_negative_distances, hard_negative_mining_distances


def compute_hard_negative_triplet_loss(
        anchor_positive_distances,
        anchor_match_distance_matrix,
        anchor_match_negative_indicator_matrix,
        margin,
        use_semi_hard,
        anchor_positive_mining_distances=None,
        anchor_match_mining_distance_matrix=None):
    """Computes triplet loss with (semi-)hard negative mining.
    Args:
      anchor_positive_distances: A tensor for anchor/positive distances. Shape =
        [num_anchors].
      anchor_match_distance_matrix: A tensor for anchor/match distance matrix.
        Shape = [num_anchors, num_matches].
      anchor_match_negative_indicator_matrix: A tensor for anchor/match negative
        indicator matrix. Shape = [num_anchors, num_matches].
      margin: A float for triplet loss margin.
      use_semi_hard: A boolean for whether to compute semi-hard negative distances
        instead of hard negative distances.
      anchor_positive_mining_distances: A tensor for positive distances of each
        anchor for (semi-)hard negative mining. Only used if `use_semi_hard` is
        True. Shape = [num_anchors].
      anchor_match_mining_distance_matrix: A tensor for an alternative
        anchor/match distance matrix to use for (semi-)hard negative mining. Use
        None to ignore and use `anchor_match_distance_matrix` instead. If
        specified, must be of the same shape as `anchor_match_distance_matrix`.

    Returns:
      loss: A tensor for loss. Shape = [].
      num_active_triplets: A tensor for number of active triplets. Shape = [].
      anchor_negative_distances: A tensor for anchor/negative distances. Shape =
        [num_amchors]. If an anchor has no (semi-)hard negative match, its
        negative distance will be assigned as the maximum value of
        anchor_match_distance_matrix.dtype.
      mining_loss: A tensor for loss based on mining distances. Shape = [].
      num_active_mining_triplets: A tensor for number of active triplets based on
        mining distances. Shape = [].
      anchor_negative_mining_distances: A tensor for anchor/negative mining
        distances. Shape = [num_amchors]. If an anchor has no (semi-)hard negative
        match, its negative distance will be assigned as the maximum value of
        anchor_match_mining_distance_matrix.dtype.

    """

    if anchor_positive_mining_distances is None:
        anchor_positive_mining_distances = anchor_positive_distances
    if anchor_match_mining_distance_matrix is None:
        anchor_match_mining_distance_matrix = anchor_match_distance_matrix

    anchor_negative_distances, anchor_negative_mining_distances = (
        compute_hard_negative_distances(
            anchor_match_distance_matrix,
            anchor_match_negative_indicator_matrix,
            use_semi_hard=use_semi_hard,
            anchor_positive_mining_distances=anchor_positive_mining_distances,
            anchor_match_mining_distance_matrix=(
                anchor_match_mining_distance_matrix)))

    def compute_triplet_loss(positive_distances, negative_distances):
        losses = torch.relu(positive_distances + margin - negative_distances)
        losses = torch.where(
            (losses < float('inf')), losses, torch.zeros_like(losses))
        num_nonzero_losses = (losses != 0).sum().item()
        loss = losses.mean()
        return loss, num_nonzero_losses

    loss, num_active_triplets = compute_triplet_loss(anchor_positive_distances,
                                                     anchor_negative_distances)
    mining_loss, num_active_mining_triplets = compute_triplet_loss(
        anchor_positive_mining_distances, anchor_negative_mining_distances)

    return (loss, num_active_triplets, anchor_negative_distances, mining_loss,
            num_active_mining_triplets, anchor_negative_mining_distances)


def compute_keypoint_triplet_losses(
        anchor_embeddings,
        positive_embeddings,
        match_embeddings,
        anchor_keypoints,   # 3d keypoints
        match_keypoints,    # 3d keypoints
        margin,
        min_negative_keypoint_distance,
        use_semi_hard,
        exclude_inactive_triplet_loss,
        anchor_keypoint_masks=None,
        match_keypoint_masks=None,
        embedding_sample_distance_fn=create_sample_distance_fn(),
        keypoint_distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
        anchor_mining_embeddings=None,
        positive_mining_embeddings=None,
        match_mining_embeddings=None,
        summarize_percentiles=True):
    """Computes triplet losses with both hard and semi-hard negatives.

    # ... [Your docstring]
    """

    def maybe_expand_sample_dim(embeddings):
        if len(embeddings.size()) == 2:
            return embeddings.unsqueeze(-2)
        return embeddings

    anchor_embeddings = maybe_expand_sample_dim(anchor_embeddings)
    positive_embeddings = maybe_expand_sample_dim(positive_embeddings)
    match_embeddings = maybe_expand_sample_dim(match_embeddings)

    # ... [Your code with TensorFlow-specific logic]

    if min_negative_keypoint_distance >= 0.0:
        anchor_match_negative_indicator_matrix = (
            compute_negative_indicator_matrix(
                anchor_points=anchor_keypoints,
                match_points=match_keypoints,
                distance_fn=keypoint_distance_fn,
                min_negative_distance=min_negative_keypoint_distance,
                anchor_point_masks=anchor_keypoint_masks,
                match_point_masks=match_keypoint_masks))
    else:
        num_anchors = anchor_keypoints.size(0)
        anchor_match_negative_indicator_matrix = torch.logical_not(
            torch.eye(num_anchors, dtype=torch.bool))

    anchor_positive_distances = embedding_sample_distance_fn(
        anchor_embeddings, positive_embeddings)

    if anchor_mining_embeddings is None and positive_mining_embeddings is None:
        anchor_positive_mining_distances = anchor_positive_distances
    else:
        anchor_positive_mining_distances = embedding_sample_distance_fn(
            anchor_embeddings if anchor_mining_embeddings is None else
            maybe_expand_sample_dim(anchor_mining_embeddings),
            positive_embeddings if positive_mining_embeddings is None else
            maybe_expand_sample_dim(positive_mining_embeddings))

    anchor_match_distance_matrix = distance_utils.compute_distance_matrix(
        anchor_embeddings,
        match_embeddings,
        distance_fn=embedding_sample_distance_fn)
    
    if anchor_mining_embeddings is None and match_mining_embeddings is None:
        anchor_match_mining_distance_matrix = anchor_match_distance_matrix
    else:
        anchor_match_mining_distance_matrix = distance_utils.compute_distance_matrix(
            anchor_embeddings if anchor_mining_embeddings is None else
            maybe_expand_sample_dim(anchor_mining_embeddings),
            match_embeddings if match_mining_embeddings is None else
            maybe_expand_sample_dim(match_mining_embeddings),
            distance_fn=embedding_sample_distance_fn)

    num_total_triplets = float(anchor_embeddings.size(0))

    def compute_loss_and_create_summaries(use_semi_hard):
        """Computes loss and creates summaries."""
        (loss, num_active_triplets, negative_distances, mining_loss,
         num_active_mining_triplets, negative_mining_distances) = (
             compute_hard_negative_triplet_loss(
                 anchor_positive_distances,
                 anchor_match_distance_matrix,
                 anchor_match_negative_indicator_matrix,
                 margin=margin,
                 use_semi_hard=use_semi_hard,
                 anchor_positive_mining_distances=anchor_positive_mining_distances,
                 anchor_match_mining_distance_matrix=(
                     anchor_match_mining_distance_matrix)))
        # print('negative_distances: ', negative_distances.shape, 'max=', negative_distances.max())
        mask = negative_distances < torch.finfo(negative_distances.dtype).max
        negative_distances = negative_distances[mask]
        # negative_distances = negative_distances[negative_distances < torch.finfo(negative_distances.dtype).max]
        # print('negative_distances: --> ', negative_distances.shape)

        mask = negative_mining_distances < torch.finfo(negative_mining_distances.dtype).max
        negative_mining_distances = negative_mining_distances[mask]
        # negative_mining_distances = negative_mining_distances[negative_distances < torch.finfo(negative_distances.dtype).max]

        active_triplet_ratio = float(num_active_triplets) / num_total_triplets
        active_mining_triplet_ratio = float(num_active_mining_triplets) / num_total_triplets

        active_loss = loss / max(1e-12, active_triplet_ratio)
        active_mining_loss = mining_loss / max(1e-12, active_mining_triplet_ratio)

        tag = 'SemiHardNegative' if use_semi_hard else 'HardNegative'
        summaries = {
            # Summaries related to triplet loss computation.
            'triplet_loss/Anchor/%s/Distance/Mean' % tag:
                torch.mean(negative_distances),
            'triplet_loss/%s/Loss/All' % tag:
                loss,
            'triplet_loss/%s/Loss/Active' % tag:
                active_loss,
            'triplet_loss/%s/ActiveTripletNum' % tag:
                num_active_triplets,
            'triplet_loss/%s/ActiveTripletRatio' % tag:
                active_triplet_ratio,

            # Summaries related to triplet mining.
            'triplet_mining/Anchor/%s/Distance/Mean' % tag:
                torch.mean(negative_mining_distances),
            'triplet_mining/%s/Loss/All' % tag:
                mining_loss,
            'triplet_mining/%s/Loss/Active' % tag:
                active_mining_loss,
            'triplet_mining/%s/ActiveTripletNum' % tag:
                num_active_mining_triplets,
            'triplet_mining/%s/ActiveTripletRatio' % tag:
                active_mining_triplet_ratio,
        }
        if summarize_percentiles:
            summaries.update({
                'triplet_loss/Anchor/%s/Distance/Median' % tag:
                    torch.quantile(negative_distances, 0.5),
                'triplet_mining/Anchor/%s/Distance/Median' % tag:
                    torch.quantile(negative_mining_distances, 0.5),
            })

        return loss, active_loss, summaries

    hard_negative_loss, hard_negative_active_loss, hard_negative_summaries = (
        compute_loss_and_create_summaries(use_semi_hard=False))
    
    (semi_hard_negative_loss, semi_hard_negative_active_loss,
     semi_hard_negative_summaries) = (
         compute_loss_and_create_summaries(use_semi_hard=True))

    summaries = {
        'triplet_loss/Margin':
            torch.tensor(margin),
        'triplet_loss/Anchor/Positive/Distance/Mean':
            torch.mean(anchor_positive_distances),
        'triplet_mining/Anchor/Positive/Distance/Mean':
            torch.mean(anchor_positive_mining_distances),
    }
    if summarize_percentiles:
        summaries.update({
            'triplet_loss/Anchor/Positive/Distance/Median':
                torch.quantile(anchor_positive_distances, 0.5),
            'triplet_mining/Anchor/Positive/Distance/Median':
                torch.quantile(anchor_positive_mining_distances, 0.5),
        })
    summaries.update(hard_negative_summaries)
    summaries.update(semi_hard_negative_summaries)

    if use_semi_hard:
        if exclude_inactive_triplet_loss:
            loss = semi_hard_negative_active_loss
        else:
            loss = semi_hard_negative_loss
    else:
        if exclude_inactive_triplet_loss:
            loss = hard_negative_active_loss
        else:
            loss = hard_negative_loss

    return loss, summaries


def compute_kl_regularization_loss(means,
                                   stddevs,
                                   loss_weight,
                                   prior_mean=0.0,
                                   prior_stddev=1.0):
    """Computes KL divergence regularization loss for multivariate Gaussian.

    Args:
      means: A tensor for distribution means. Shape = [..., dim].
      stddevs: A tensor for distribution standard deviations. Shape = [..., dim].
      loss_weight: A float for loss weight.
      prior_mean: A float for prior distribution mean.
      prior_stddev: A float for prior distribution standard deviation.

    Returns:
      loss: A tensor for weighted regularization loss. Shape = [].
      summaries: A dictionary for loss summaries.
    """
    loss = torch.mean(
        distance_utils.compute_gaussian_kl_divergence(
            means, stddevs, rhs_means=torch.tensor(prior_mean), rhs_stddevs=torch.tensor(prior_stddev)))
    weighted_loss = loss_weight * loss
    summaries = {
        'regularization_loss/KL/PriorMean/Mean':
            torch.mean(torch.tensor(prior_mean)),
        'regularization_loss/KL/PriorVar/Mean':
            torch.mean(torch.tensor(prior_stddev)**2),
        'regularization_loss/KL/Loss/Original':
            loss,
        'regularization_loss/KL/Loss/Weighted':
            weighted_loss,
        'regularization_loss/KL/Loss/Weight':
            torch.tensor(loss_weight),
    }
    return weighted_loss, summaries

def compute_positive_pairwise_loss(anchor_embeddings,
                                   positive_embeddings,
                                   loss_weight,
                                   distance_fn=functools.partial(
                                       distance_utils.compute_l2_distances,
                                       squared=True)):
    """Computes anchor/positive pairwise (squared L2) loss.

    Args:
      anchor_embeddings: A tensor for anchor embeddings. Shape = [...,
        embedding_dim].
      positive_embeddings: A tensor for positive embeddings. Shape = [...,
        embedding_dim].
      loss_weight: A float for loss weight.
      distance_fn: A function handle for computing embedding distances, which
        takes two embedding tensors of shape [..., embedding_dim] and returns a
        distance tensor of shape [...].

    Returns:
      loss: A tensor for weighted positive pairwise loss. Shape = [].
      summaries: A dictionary for loss summaries.
    """
    loss = torch.mean(
        distance_fn(anchor_embeddings, positive_embeddings))
    weighted_loss = loss_weight * loss
    summaries = {
        'pairwise_loss/PositivePair/Loss/Original': loss,
        'pairwise_loss/PositivePair/Loss/Weighted': weighted_loss,
        'pairwise_loss/PositivePair/Loss/Weight': torch.tensor(loss_weight),
    }
    return weighted_loss, summaries

"""
This function is used for CPU which is equaivalient to the training CUDA version
Note that we use this metric for ANN search, so we flatten the [k, 16] into [k*16] vector,
Here we should reshape the [k*16] vector into [k, 16]

Using Numpy broadcasting technique speedup at least 100 times than double for loops.
"""
def probabilistic_distance(A, B, sigmoid_a=1, sigmoid_b=0):
    def scaled_sigmoid(inner_distances, sigmoid_a, sigmoid_b, smoothing=0.1):
        z = sigmoid_a * inner_distances - sigmoid_b
        z = np.clip(z, -20, 20)  # max_value could be something like 10 or 20
        p = 1 / (1 + np.exp(z))    
        # p = 1 / (1 + np.exp(sigmoid_a * inner_distances - sigmoid_b))
        if smoothing > 0.0:
            p = (1.0 - smoothing) * p + smoothing / 2.0
        return p    
    # Reshape A and B to [..., 20, 16]
    A_reshaped = A.reshape(*A.shape[:-1], 20, 16)
    B_reshaped = B.reshape(*B.shape[:-1], 20, 16)    
    # Compute pairwise L2 distances
    distances = np.linalg.norm(A_reshaped - B_reshaped, axis=-1)    

    # Apply the scaled sigmoid function
    distances = scaled_sigmoid(distances, sigmoid_a, sigmoid_b)
    
    return -np.log(np.mean(distances, axis=-1))


""" This function will be called in PyTorch, I will test after the Numpy version integration
"""
def probabilistic_distance_torch(A, B, sigmoid_a=5.24, sigmoid_b=11.16, smoothing=0.1):
    def scaled_sigmoid(distances, sigmoid_a, sigmoid_b, smoothing=0.1):
        z = sigmoid_a * distances - sigmoid_b
        z = torch.clamp(z, -20, 20)
        p = 1 / (1 + torch.exp(z))
        if smoothing > 0.0:
            p = (1.0 - smoothing) * p + smoothing / 2.0
        return p

    # Reshape A and B to [..., 20, 16] and expand one dimension for broadcasting
    A_reshaped = A.reshape(*A.shape[:-1], 20, 1, 16)
    B_reshaped = B.reshape(*B.shape[:-1], 1, 20, 16)

    # Compute pairwise L2 distances
    distances = torch.norm(A_reshaped - B_reshaped, dim=-1)

    # Apply the scaled sigmoid function
    distances = scaled_sigmoid(distances, sigmoid_a, sigmoid_b, smoothing)
    distances =  -torch.log(distances.mean(dim=[-1, -2])) 

    return distances
