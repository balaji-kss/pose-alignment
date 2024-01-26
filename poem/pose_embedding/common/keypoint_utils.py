import torch
import numpy as np

from pose_embedding.common import distance_utils
from pose_embedding.common import data_utils


def get_points(points, indices):
    """Gets points as the centers of points at specified indices.

    Args:
      points: A tensor for points. Shape = [..., num_points, point_dim].
      indices: A list of integers for point indices.

    Returns:
      A tensor for (center) points. Shape = [..., 1, point_dim].

    Raises:
      ValueError: If `indices` is empty.
    """
    if not indices:
        raise ValueError('`Indices` must be non-empty.')

    # Using torch's index_select for gathering indices
    points = torch.index_select(points, dim=-2, index=torch.tensor(indices))

    if len(indices) == 1:
        return points

    # Using torch's mean function to compute the mean along the specified axis
    return torch.mean(points, dim=-2, keepdim=True)


def swap_x_y(points):
    """Swaps the order of the first two dimensions (x and y) coordinate.

    Args:
      points: A tensor for points. Shape = [..., point_dim].

    Returns:
      A tensor for points with swapped x and y.

    Raises:
      ValueError: If point dimension is less than 2.
    """
    point_dim = points.shape[-1]
    if point_dim < 2:
        raise ValueError(
            'Point dimension must be greater than or equal to 2: %d.' % point_dim)

    perm_indices = [1, 0] + list(range(2, point_dim))

    # Using torch's index_select for swapping x and y
    return points[..., perm_indices]


def normalize_points(points, offset_point_indices,
                     scale_distance_point_index_pairs,
                     scale_distance_reduction_fn, scale_unit):
    offset_points = get_points(points, offset_point_indices)

    def compute_scale_distances():
        sub_scale_distances_list = []
        for lhs_indices, rhs_indices in scale_distance_point_index_pairs:
            lhs_points = get_points(points, lhs_indices)
            rhs_points = get_points(points, rhs_indices)
            sub_scale_distances_list.append(
                distance_utils.compute_l2_distances(lhs_points, rhs_points, keepdims=True))

        # Using torch.cat to concatenate tensors along a specified dimension
        sub_scale_distances = torch.cat(sub_scale_distances_list, dim=-1)
        return scale_distance_reduction_fn(sub_scale_distances, dim=-1, keepdim=True)

    scale_distances = torch.maximum(compute_scale_distances(), torch.tensor(1e-12))
    normalized_points = (points - offset_points) / scale_distances * scale_unit
    return normalized_points, offset_points, scale_distances


def centralize_masked_points(points, point_masks):
    """Sets masked out points to the centers of the rest of the points.

    Args:
      points: A tensor for points. Shape = [..., num_points, point_dim].
      point_masks: A tensor for the masks. Shape = [..., num_points].

    Returns:
      A tensor for points with masked out points centralized.
    """
    point_masks_expanded = point_masks.unsqueeze(-1)
    kept_centers = data_utils.reduce_weighted_mean(
        points, weights=point_masks_expanded, axis=-2, keepdims=True)
    return torch.where(point_masks_expanded.to(dtype=torch.bool), points, kept_centers)


def standardize_points(points):
    """Standardizes points by centering and scaling.

    Args:
      points: A tensor for input points. Shape = [..., num_points, point_dim].

    Returns:
      standardized_points: A tensor for standardized points. Shape = [..., num_points,
        point_dim].
      offsets: A tensor for the applied offsets. Shape = [..., 1, point_dim].
      scales: A tensor for the applied inverse scales. Shape = [..., 1,
        point_dim].
    """
    offsets = points.mean(dim=-2, keepdim=True)
    standardized_points = points - offsets
    scales = torch.sqrt(
        (standardized_points**2).sum(dim=(-2, -1), keepdim=True))
    standardized_points = standardized_points / scales
    return standardized_points, offsets, scales


# def standardize_points(points):
#     """Standardizes points by centering and scaling.

#     Args:
#       points: A tensor for input points. Shape = [..., num_points, point_dim].

#     Returns:
#       points: A tensor for standardized points. Shape = [..., num_points,
#         point_dim].
#       offsets: A tensor for the applied offsets. Shape = [..., 1, point_dim].
#       scales: A tensor for the applied inverse scales. Shape = [..., 1,
#         point_dim].
#     """
#     offsets = points.mean(dim=-2, keepdim=True)
#     points -= offsets
#     scales = torch.sqrt((points**2).sum(dim=(-2, -1), keepdim=True))
#     points /= scales
#     return points, offsets, scales


def compute_procrustes_alignment_params(target_points, source_points, point_masks=None):
    def tile_last_dims_pytorch(x, last_dim_multiples):
        """Tiles the last dimensions of a tensor in PyTorch."""
        multiples = [1] * (x.dim() - len(last_dim_multiples)
                           ) + list(last_dim_multiples)
        return x.repeat(multiples)

    """Computes Procrustes alignment parameters using PyTorch."""

    if point_masks is not None:
        target_points = centralize_masked_points(target_points, point_masks)
        source_points = centralize_masked_points(source_points, point_masks)

    # Standardize the target and source points
    standardized_target_points, target_offsets, target_scales = standardize_points(
        target_points)
    standardized_source_points, source_offsets, source_scales = standardize_points(
        source_points)

    # Compute the 'a' matrix
    a = torch.matmul(standardized_target_points.transpose(-2, -1),
                     standardized_source_points)

    # Compute the Singular Value Decomposition
    u, s, vt = torch.linalg.svd(a)

    # Compute rotation matrix 'r'
    r = torch.matmul(vt, u)

    # Correct for potential reflections (to ensure a right-handed system)
    det_r = torch.det(r)
    signs = torch.sign(det_r)
    signs = signs.unsqueeze(-1)

    point_dim = target_points.shape[-1]
    ones = torch.ones_like(signs)
    ones = tile_last_dims_pytorch(ones, [point_dim - 1])
    signs = torch.cat([ones, signs], dim=-1)

    s *= signs
    signs = signs.unsqueeze(-2)
    # vt *= signs
    v = vt.transpose(-2, -1) * signs

    # Recalculate rotation after reflection correction
    rotations = torch.matmul(v, u.transpose(-2, -1))
    # rotations = torch.matmul(v.transpose(-2, -1), u.transpose(-2, -1))
    scales = s.sum(dim=-1, keepdim=True).unsqueeze(-1) * \
        target_scales / source_scales

    # Calculate translations
    translations = target_offsets - scales * \
        torch.matmul(source_offsets, rotations)

    return rotations, scales, translations


def procrustes_align_points(target_points, source_points, point_masks=None):
    """Performs Procrustes alignment on source points to target points using PyTorch.

    Args:
      target_points: A tensor for target points. Shape = [..., num_points, point_dim].
      source_points: A tensor for source points. Shape = [..., num_points, point_dim].
      point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if None.

    Returns:
      A tensor for aligned source points. Shape = [..., num_points, point_dim].
    """

    rotations, scales, translations = compute_procrustes_alignment_params(
        target_points, source_points, point_masks=point_masks)
    return translations + scales * torch.matmul(source_points, rotations)


def compute_mpjpes(lhs_points, rhs_points, point_masks=None):
    """Computes the Mean Per-Joint Position Errors (MPJPEs).

    If `point_masks` is specified, computes MPJPEs weighted by `point_masks`.

    Args:
      lhs_points: A tensor for the LHS points. Shape = [..., num_points,
        point_dim].
      rhs_points: A tensor for the RHS points. Shape = [..., num_points,
        point_dim].
      point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
        None.

    Returns:
      A tensor for MPJPEs. Shape = [...].
    """
    distances = distance_utils.compute_l2_distances(
        lhs_points, rhs_points, keepdims=False)
    return data_utils.reduce_weighted_mean(
        distances, weights=point_masks, axis=-1)


def compute_procrustes_aligned_mpjpes(target_points,
                                      source_points,
                                      point_masks=None):
    """Computes MPJPEs after Procrustes alignment.

    Args:
      target_points: A tensor for target points. Shape = [..., num_points,
        point_dim].
      source_points: A tensor for source points. Shape = [..., num_points,
        point_dim].
      point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
        None.

    Returns:
      A tensor for MPJPEs. Shape = [...].
    """
    aligned_source_points = procrustes_align_points(
        target_points, source_points, point_masks=point_masks)

    return compute_mpjpes(
        aligned_source_points, target_points, point_masks=point_masks)


def denormalize_points_by_image_size(points, image_sizes):
    """Denormalizes point coordinates by image sizes.

    Args:
      points (torch.Tensor): Normalized points by image size. Shape = [...,
        num_points, point_dim].
      image_sizes (torch.Tensor): Image sizes. Shape = [..., point_dim].

    Returns:
      torch.Tensor: Denormalized points. Shape = [..., num_points, point_dim].
    """
    if len(image_sizes.shape) != len(points.shape) - 1:
        raise ValueError(
            f'Rank of `image_sizes` must be that of `points` minus 1: {len(image_sizes.shape)} vs. {len(points.shape)}.')

    return points * image_sizes.unsqueeze(-2).to(points.dtype)


def select_keypoints_by_name(keypoints, input_keypoint_names, output_keypoint_names, keypoint_masks=None):
    """Selects keypoints by name.

    Note that it is the user's responsibility to ensure that the output keypoint
    name list is a subset of the input keypoint names.

    Args:
      keypoints (torch.Tensor): Input keypoints. Shape = [..., num_input_keypoints, point_dim].
      input_keypoint_names (List[str]): A list of strings for input keypoint names.
      output_keypoint_names (List[str]): A list of strings for output keypoint names.
      keypoint_masks (torch.Tensor, optional): Input keypoint masks. Shape = [..., num_input_keypoints]. None if not provided.

    Returns:
      output_keypoints (torch.Tensor): Output keypoints. Shape = [..., num_output_keypoints, point_dim].
      output_keypoint_masks (torch.Tensor, optional): Output keypoint masks. Shape = [..., num_output_keypoints]. None if input mask tensor is None.
    """
    input_to_output_indices = torch.tensor([
        input_keypoint_names.index(keypoint_name)
        for keypoint_name in output_keypoint_names
    ], dtype=torch.long)

    output_keypoints = torch.index_select(
        keypoints, dim=-2, index=input_to_output_indices)

    output_keypoint_masks = None
    if keypoint_masks is not None:
        output_keypoint_masks = torch.index_select(
            keypoint_masks, dim=-1, index=input_to_output_indices)

    return output_keypoints, output_keypoint_masks


def create_rotation_matrices_3d(azimuths, elevations, rolls):
    """Creates rotation matrices given rotation angles.

    Note that the created rotations are to be applied on points with layout (x, y,
    z).

    Args:
      azimuths: A tensor for azimuths angles. Shape = [...].
      elevations: A tensor for elevation angles. Shape = [...].
      rolls: A tensor for roll angles. Shape = [...].

    Returns:
      A tensor for rotation matrices. Shape = [..., 3, 3].
    """
    azi_cos = torch.cos(azimuths)
    azi_sin = torch.sin(azimuths)
    ele_cos = torch.cos(elevations)
    ele_sin = torch.sin(elevations)
    rol_cos = torch.cos(rolls)
    rol_sin = torch.sin(rolls)
    rotations_00 = azi_cos * ele_cos
    rotations_01 = azi_cos * ele_sin * rol_sin - azi_sin * rol_cos
    rotations_02 = azi_cos * ele_sin * rol_cos + azi_sin * rol_sin
    rotations_10 = azi_sin * ele_cos
    rotations_11 = azi_sin * ele_sin * rol_sin + azi_cos * rol_cos
    rotations_12 = azi_sin * ele_sin * rol_cos - azi_cos * rol_sin
    rotations_20 = -ele_sin
    rotations_21 = ele_cos * rol_sin
    rotations_22 = ele_cos * rol_cos
    rotations_0 = torch.stack(
        [rotations_00, rotations_10, rotations_20], axis=-1)
    rotations_1 = torch.stack(
        [rotations_01, rotations_11, rotations_21], axis=-1)
    rotations_2 = torch.stack(
        [rotations_02, rotations_12, rotations_22], axis=-1)
    return torch.stack([rotations_0, rotations_1, rotations_2], axis=-1)


def create_interpolated_rotation_matrix_sequences(start_euler_angles,
                                                  end_euler_angles,
                                                  sequence_length):
    """Creates parameters to generate interpolated rotation trajectories.

    Args:
      start_euler_angles: A tuple for start Euler angles including (azimuths,
        elevations, rolls). Each euler angle tensor in the tuple is in shape
        [...].
      end_euler_angles: A tuple for end Euler angles including (azimuths,
        elevations, rolls). Each euler angle tensor in the tuple is in shape
        [...].
      sequence_length: An integer for length of sequence to rotate.

    Returns:
      rotations: A rotation metrics of shape [..., sequence_length, 3, 3].

    Raises:
      ValueError: Sequence length is smaller than 2.
    """
    if sequence_length < 2:
        raise ValueError('Minimal number of views needs to be at least 2.')

    division = sequence_length - 1
    start_azimuths, start_elevations, start_rolls = start_euler_angles
    end_azimuths, end_elevations, end_rolls = end_euler_angles
    delta_azimuths = (end_azimuths - start_azimuths) / division
    delta_elevations = (end_elevations - start_elevations) / division
    delta_rolls = (end_rolls - start_rolls) / division

    rotations = []
    for i in range(sequence_length):
        azimuths = start_azimuths + delta_azimuths * i
        elevations = start_elevations + delta_elevations * i
        rolls = start_rolls + delta_rolls * i
        rotations.append(create_rotation_matrices_3d(
            azimuths, elevations, rolls))
    rotations = torch.stack(rotations, axis=-3)
    return rotations


def randomly_rotate_3d(keypoints_3d, azimuth_range, elevation_range, roll_range, sequential_inputs=False, seed=None):
    """Randomly rotates 3D keypoints.

    Args:
      keypoints_3d (torch.Tensor): A tensor for 3D keypoints. Shape = [..., num_keypoints, 3].
      azimuth_range (tuple): Range for azimuth rotation.
      elevation_range (tuple): Range for elevation rotation.
      roll_range (tuple): Range for roll rotation.
      sequential_inputs (bool): If keypoints are sequential.
      seed (int, optional): Random seed.

    Returns:
      torch.Tensor: Rotated keypoints_3d.
    """
    def create_random_angles(angle_range):
        """Creates random starting and ending camera angles."""
        if len(angle_range) == 2:
            shape = keypoints_3d.shape[:-3]
            start_angles = torch.rand(
                *shape) * (angle_range[1] - angle_range[0]) + angle_range[0]
            end_angles = start_angles
        elif len(angle_range) == 4:
            shape = keypoints_3d.shape[:-3]
            start_angles = torch.rand(
                *shape) * (angle_range[1] - angle_range[0]) + angle_range[0]
            angle_deltas = torch.rand(
                *shape) * (angle_range[3] - angle_range[2]) + angle_range[2]
            end_angles = start_angles + angle_deltas
        else:
            raise ValueError(
                'Unsupported angle range: `{}`.'.format(angle_range))
        return start_angles, end_angles

    if seed is not None:
        torch.manual_seed(seed)

    if not sequential_inputs:
        shape = keypoints_3d.shape[:-2]
        if shape:
            azimuths = torch.rand(
                *shape) * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
            elevations = torch.rand(
                *shape) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]
            rolls = torch.rand(*shape) * \
                (roll_range[1] - roll_range[0]) + roll_range[0]
        else:
            azimuths = torch.rand(
                1) * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
            elevations = torch.rand(
                1) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]
            rolls = torch.rand(
                1) * (roll_range[1] - roll_range[0]) + roll_range[0]

        rotation_matrices = create_rotation_matrices_3d(
            azimuths, elevations, rolls)
    else:
        sequence_length = keypoints_3d.shape[-3]

        start_azimuths, end_azimuths = create_random_angles(azimuth_range)
        start_elevations, end_elevations = create_random_angles(
            elevation_range)
        start_rolls, end_rolls = create_random_angles(roll_range)

        start_euler_angles = (start_azimuths, start_elevations, start_rolls)
        end_euler_angles = (end_azimuths, end_elevations, end_rolls)

        rotation_matrices = create_interpolated_rotation_matrix_sequences(
            start_euler_angles, end_euler_angles, sequence_length=sequence_length)

    # Swap x and y dimensions before rotation and swap back afterwards.
    # print("keypoints_3d_shape:\n", keypoints_3d.numpy().shape)
    # print("rotation_matrices_shape:\n", rotation_matrices.numpy().shape)

    def transpose_last_two_dims(tensor):
        # permute dims except for the last two, which are swapped
        return tensor.permute(*range(tensor.ndim-2), -1, -2)

    keypoints_3d = swap_x_y(keypoints_3d)
    keypoints_3d_transposed = transpose_last_two_dims(keypoints_3d)
    result = torch.matmul(rotation_matrices, keypoints_3d_transposed)
    keypoints_3d = transpose_last_two_dims(result)
    # keypoints_3d = torch.matmul(rotation_matrices, keypoints_3d.permute(*range(keypoints_3d.ndim-2), -1, -2)).permute(*range(keypoints_3d.ndim-2), -1, -2)
    return swap_x_y(keypoints_3d)


def randomly_rotate_and_project_3d_to_2d(keypoints_3d,
                                         azimuth_range,
                                         elevation_range,
                                         roll_range,
                                         normalized_camera_depth_range,
                                         sequential_inputs=False,
                                         seed=None):
    """Randomly rotates and projects 3D keypoints to 2D."""

    keypoints_3d = randomly_rotate_3d(
        keypoints_3d,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        roll_range=roll_range,
        sequential_inputs=sequential_inputs,
        seed=seed)

    # Transform to default camera coordinate.
    default_rotation_to_camera = torch.tensor([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    keypoints_3d = torch.matmul(
        default_rotation_to_camera, keypoints_3d.transpose(-1, -2)).transpose(-1, -2)

    # Move to default depth.
    if sequential_inputs:
        shape = keypoints_3d.shape[:-3]
        normalized_camera_depths = torch.rand(
            *shape) * (normalized_camera_depth_range[1] - normalized_camera_depth_range[0]) + normalized_camera_depth_range[0]
        normalized_camera_depths = normalized_camera_depths.unsqueeze(
            -1).unsqueeze(-1)
    else:
        shape = keypoints_3d.shape[:-2]
        if shape:
            normalized_camera_depths = torch.rand(
                *shape) * (normalized_camera_depth_range[1] - normalized_camera_depth_range[0]) + normalized_camera_depth_range[0]
        else:
            normalized_camera_depths = torch.rand(
                1) * (normalized_camera_depth_range[1] - normalized_camera_depth_range[0]) + normalized_camera_depth_range[0]
        normalized_camera_depths = normalized_camera_depths.unsqueeze(-1)

    default_centers = torch.stack([
        torch.zeros_like(normalized_camera_depths),
        torch.zeros_like(normalized_camera_depths),
        normalized_camera_depths,
    ], dim=-1)
    keypoints_3d += default_centers

    # Project to 2D.
    if keypoints_3d.shape[0] == 1 and len(keypoints_3d.shape) == 3:
        keypoints_3d.squeeze_(0)
    return keypoints_3d[..., :-1] / torch.maximum(torch.tensor(1e-12).to(keypoints_3d.device), keypoints_3d[..., -1:])


def randomly_project_and_select_keypoints(keypoints_3d,
                                          keypoint_profile_3d,
                                          output_keypoint_names,
                                          azimuth_range,
                                          elevation_range,
                                          roll_range,
                                          normalized_camera_depth_range,
                                          keypoint_masks_3d=None,
                                          normalize_before_projection=True,
                                          sequential_inputs=False,
                                          seed=None):
    """Generates 2D keypoints from random 3D keypoint projection.

    Notes:
    1. The default camera z will be added to the keypoint depths before
       projection, which underlyingly assumes the input 3D keypoints are centered
       at camera origin (`normalize_before_projection` defaults to True).
    2. The compatible 3D keypoint names (if specified during 2D keypoint profile
       initialization) will be used for the 2D keypoint profile.

    Args:
      keypoints_3d: A tensor for input 3D keypoints. Shape = [...,
        num_keypoints_3d, 3].
      keypoint_profile_3d: A KeypointProfile3D object for input keypoints.
      output_keypoint_names: A list of keypoint names to select 2D projection
        with. Must be a subset of the 3D keypoint names.
      azimuth_range: A 2-tuple for minimum and maximum azimuth angles to randomly
        rotate 3D keypoints with. For sequential inputs, also supports 4-tuple for
        minimum/maximum angles as well as minimum/maximum angle deltas between
        starting and ending angles.
      elevation_range: A 2-tuple for minimum and maximum elevation angles to
        randomly rotate 3D keypoints with. For sequential inputs, also supports
        4-tuple for minimum/maximum angles as well as minimum/maximum angle deltas
        between starting and ending angles.
      roll_range: A 2-tuple for minimum and maximum roll angles to randomly rotate
        3D keypoints with. For sequential inputs, also supports 4-tuple for
        minimum/maximum angles as well as minimum/maximum angle deltas between
        starting and ending angles.
      normalized_camera_depth_range: A tuple for minimum and maximum normalized
        camera depth for random camera augmentation.
      keypoint_masks_3d: A tensor for input 3D keypoint masks. Shape = [...,
        num_keypoints_3d]. Ignored if None.
      normalize_before_projection: A boolean for whether to normalize 3D poses
        before projection.
      sequential_inputs: A boolean flag indicating whether the inputs are
        sequential, if true, the input keypoints are supposed to be in shape [...,
        sequence_length, num_keypoints, 3].
      seed: An integer for random seed.

    Returns:
      keypoints_2d: A tensor for output 2D keypoints. Shape = [...,
        num_keypoints_2d, 2].
      keypoint_masks_2d: A tensor for output 2D keypoint masks. Shape = [...,
        num_keypoints_2d]. None if input 3D mask is not specified.

    Raises:
      ValueError: If keypoint profile has unsupported dimensionality.
    """
    if keypoint_profile_3d.keypoint_dim != 3:
        raise ValueError('Unsupported input keypoint dimension: %d.' %
                         keypoint_profile_3d.keypoint_dim)

    if normalize_before_projection:
        keypoints_3d, _, _ = (
            keypoint_profile_3d.normalize(keypoints_3d, keypoint_masks_3d))

    # Keep 3D keypoint masks as 2D masks.
    keypoints_3d, keypoint_masks_2d = select_keypoints_by_name(
        keypoints_3d,
        input_keypoint_names=keypoint_profile_3d.keypoint_names,
        output_keypoint_names=output_keypoint_names,
        keypoint_masks=keypoint_masks_3d)
    keypoints_2d = randomly_rotate_and_project_3d_to_2d(
        keypoints_3d,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        roll_range=roll_range,
        normalized_camera_depth_range=normalized_camera_depth_range,
        sequential_inputs=sequential_inputs,
        seed=seed)
    return keypoints_2d, keypoint_masks_2d


def transfer_keypoint_masks(input_keypoint_masks,
                            input_keypoint_profile,
                            output_keypoint_profile,
                            enforce_surjectivity=False):
    """Transfers keypoint masks according to a different profile.

    Args:
      input_keypoint_masks: A list of tensors for input keypoint masks.
      input_keypoint_profile: A KeypointProfile object for input keypoints.
      output_keypoint_profile: A KeypointProfile object for output keypoints.
      enforce_surjectivity: A boolean for whether to enforce all output keypoint
        masks are transferred from input keypoint masks. If True and any output
        keypoint mask does not come from some input keypoint mask, error will be
        raised. If False, uncorresponded output keypoints will have all-one masks.

    Returns:
      A tensor for output keypoint masks.

    Raises:
      ValueError: `Enforce_surjective` is True, but mapping from input keypoint to
        output keypoint is not surjective.
    """
    input_keypoint_masks = torch.split(
        input_keypoint_masks,
        split_size_or_sections=1,
        dim=-1
    )
    output_keypoint_masks = [None] * output_keypoint_profile.keypoint_num

    for part_name in input_keypoint_profile.standard_part_names:
        input_keypoint_index = input_keypoint_profile.get_standard_part_index(
            part_name)
        output_keypoint_index = output_keypoint_profile.get_standard_part_index(
            part_name)

        if len(output_keypoint_index) != 1:
            continue

        if len(input_keypoint_index) == 1:
            output_keypoint_masks[output_keypoint_index[0]
                                  ] = input_keypoint_masks[input_keypoint_index[0]]
        else:
            input_keypoint_mask_subset = [
                input_keypoint_masks[i] for i in input_keypoint_index]
            output_keypoint_masks[output_keypoint_index[0]] = torch.prod(
                torch.stack(input_keypoint_mask_subset, dim=-1),
                dim=-1
            )

    for i, output_keypoint_mask in enumerate(output_keypoint_masks):
        if output_keypoint_mask is None:
            if enforce_surjectivity:
                raise ValueError(
                    f'Uncorresponded output keypoints: index = {i}')
            else:
                # output_keypoint_masks[i] = torch.ones_like(input_keypoint_masks[0])
                output_keypoint_masks[i] = torch.zeros_like(
                    input_keypoint_masks[0])

    return torch.cat(output_keypoint_masks, dim=-1)


def apply_stratified_instance_keypoint_dropout(keypoint_masks,
                                            probability_to_apply,
                                            probability_to_drop):
    """Applies stratified keypoint dropout on each instance
    """

    if probability_to_apply <= 0.0 or probability_to_drop <= 0.0:
        raise ValueError('Invalid dropout probabilities: (%f, %f)' %
                        (probability_to_apply, probability_to_drop))

    # Instance-level Dropout:
    keep_instance_chances = torch.rand(*keypoint_masks.shape[:-1])
    drop_instance_masks = (keep_instance_chances < probability_to_apply).unsqueeze(-1)

    # Keypoint-level Dropout:
    keep_keypoint_chances = torch.rand(*keypoint_masks.shape)
    drop_keypoint_masks = keep_keypoint_chances < probability_to_drop

    # Combining the Dropouts:
    drop_masks = drop_instance_masks & drop_keypoint_masks
    keep_masks = ~drop_masks

    return torch.where(keep_masks, keypoint_masks, torch.zeros_like(keypoint_masks))

