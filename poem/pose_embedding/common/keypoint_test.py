import torch
import numpy as np
import unittest
import math
from pose_embedding.common import keypoint_utils
from pose_embedding.common import keypoint_profiles
from dataset import pose_pairs_dataset


class TestOperations(unittest.TestCase):

    def assertTensorClose(self, a, b, atol=1e-5, rtol=1e-4):
        """
        Custom assert function to check if two tensors are close
        """
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol),
                        f"Tensors are not close: \n{a} \n{b}")

    def test_denormalize_points_by_image_size(self):
        points = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
        ])
        image_sizes = torch.tensor([[100, 200], [20, 10]])
        denormalized_points = keypoint_utils.denormalize_points_by_image_size(
            points, image_sizes)
        self.assertTensorClose(denormalized_points, torch.tensor([
            [[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]],
            [[0.2, 0.2], [0.6, 0.4], [1.0, 0.6]],
            ]))

    def test_compute_procrustes_aligned_mpjpes_case(self):
        # case 6: rank-deficient matrix
        target_points = torch.tensor([[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                 [2.5, 1.3, 1.4]])
        source_points = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTensorClose(mpjpes,  torch.tensor(0.3496029))     

        # case 0: full rank
        target_points = torch.tensor([[[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                       [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]],
                                      [[1.0, 0.0, 1.0], [5.0, 2.0, 2.0],
                                       [4.0, 0.0, 0.0], [-1.0, 3.0, -2.0]],
                                      [[2.0, 0.0, 2.0], [10.0, 4.0, 4.0],
                                       [8.0, 0.0, 0.0], [-2.0, 6.0, -4.0]]])
        source_points = torch.tensor([[[3.0, 1.0, 5.0], [-2.0, 3.0, 0.0],
                                       [1.0, -1.0, 1.0], [8.0, 3.0, -2.0]],
                                      [[3.0, 5.0, 1.0], [-2.0, 0.0, 3.0],
                                       [1.0, 1.0, -1.0], [8.0, -2.0, 3.0]],
                                      [[6.0, 10.0, 2.0], [-4.0, 0.0, 6.0],
                                       [2.0, 2.0, -2.0], [16.0, -4.0, 6.0]]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTrue(torch.allclose(mpjpes, torch.tensor([0.63214386, 0.63214344, 1.2642869 ]), atol=1e-4))   
        
        # case 2: full rank
        target_points = torch.tensor([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                     [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
        source_points = torch.tensor([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                     [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTensorClose(mpjpes, torch.tensor(0.0))

        # case 3: full rank
        target_points = torch.tensor([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
        source_points = torch.tensor([[1.5, 0.5, 0.0], [5.0, 2.0, 2.2],
                                 [4.1, 0.0, -1.0], [-1.0, -2.5, -2.0]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTensorClose(mpjpes, torch.tensor(1.00227))

        # case 4: full rank
        target_points = torch.tensor([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
        source_points = torch.tensor([[-10.0, -24.5, -49.5], [-9.0, -22.5, -49.0],
                                 [-10.0, -23.0, -50.0], [-8.5, -25.5, -51.0]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTensorClose(mpjpes, torch.tensor(0.0))                                


        # case 5: rank-deficient matrix
        target_points = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0]])
        source_points = torch.tensor([[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                 [2.5, 1.3, 1.4]])
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTensorClose(mpjpes,  torch.tensor(0.133016))  

        # case 1: rank-deficient matrix
        target_points = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]],
                                    [[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                    [2.5, 1.3, 1.4]]])
        source_points = torch.tensor([[[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                    [2.5, 1.3, 1.4]],
                                    [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]]])        
        mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
                        target_points, source_points)
        self.assertTrue(torch.allclose(mpjpes, torch.tensor([0.133016, 0.3496029]), atol=1e-4))          



    def test_compute_procrustes_alignment_params(self):
        # Shape = [3, 4, 3].
        target_points = torch.tensor([[[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                       [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]],
                                      [[1.0, 0.0, 1.0], [5.0, 2.0, 2.0],
                                       [4.0, 0.0, 0.0], [-1.0, 3.0, -2.0]],
                                      [[2.0, 0.0, 2.0], [10.0, 4.0, 4.0],
                                       [8.0, 0.0, 0.0], [-2.0, 6.0, -4.0]]])
        source_points = torch.tensor([[[3.0, 1.0, 5.0], [-2.0, 3.0, 0.0],
                                       [1.0, -1.0, 1.0], [8.0, 3.0, -2.0]],
                                      [[3.0, 5.0, 1.0], [-2.0, 0.0, 3.0],
                                       [1.0, 1.0, -1.0], [8.0, -2.0, 3.0]],
                                      [[6.0, 10.0, 2.0], [-4.0, 0.0, 6.0],
                                       [2.0, 2.0, -2.0], [16.0, -4.0, 6.0]]])
        
        rotations, scales, translations = keypoint_utils.compute_procrustes_alignment_params(target_points, source_points)
        
        expected_rotations = torch.tensor([[[-0.87982, -0.47514731, 0.01232074],
                                            [-0.31623112, 0.60451691, 0.73113418],
                                            [-0.35484453, 0.63937027, -0.68212243]],
                                           [[-0.87982, 0.01232074, -0.47514731],
                                            [-0.35484453, -0.68212243, 0.63937027],
                                            [-0.31623112, 0.73113418, 0.60451691]],
                                           [[-0.87982, 0.01232074, -0.47514731],
                                            [-0.35484453, -0.68212243, 0.63937027],
                                            [-0.31623112, 0.73113418, 0.60451691]]])
        
        expected_scales = torch.tensor([[[0.63716284347]], 
                                        [[0.63716284347]], 
                                        [[0.63716284347]]])
        
        expected_translations = torch.tensor([[[4.17980137, 0.02171898, 0.96621997]],
                                              [[4.17980137, 0.96621997, 0.02171898]],
                                              [[8.35960274, 1.93243994, 0.04343796]]])

        self.assertTrue(torch.allclose(scales, expected_scales, atol=1e-4))
        self.assertTrue(torch.allclose(translations, expected_translations, atol=1e-4))
        self.assertTrue(torch.allclose(rotations, expected_rotations, atol=1e-4))


    def test_standardize_points(self):
        # Shape = [2, 3, 2].
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                          [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]])
        standardized_x, offsets, scales = keypoint_utils.standardize_points(x)
        self.assertTensorClose(standardized_x,
                       torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]],
                                     [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]]))
        self.assertTensorClose(offsets, torch.tensor([[[3.0, 4.0]], [[6.0, 8.0]]]))
        self.assertTensorClose(scales, torch.tensor([[[4.0]], [[8.0]]]))

    def test_normalize_points(self):
        # Shape = [2, 1, 3, 2].
        points = torch.tensor([[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
                               [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]])
        offset_point_indices = [0, 1]
        scale_distance_point_index_pairs = [([0, 1], [1]), ([0], [1, 2])]
        normalized_points, offset_points, scale_distances = keypoint_utils.normalize_points(
            points,
            offset_point_indices=offset_point_indices,
            scale_distance_point_index_pairs=scale_distance_point_index_pairs,
            scale_distance_reduction_fn=torch.sum,
            scale_unit=1.0)

        sqrt_2 = 1.414213562
        expected_normalized_points = torch.tensor([
            [[
                [-0.25 / sqrt_2, -0.25 / sqrt_2],
                [0.25 / sqrt_2, 0.25 / sqrt_2],
                [0.75 / sqrt_2, 0.75 / sqrt_2],
            ]],
            [[
                [-0.25 / sqrt_2, -0.25 / sqrt_2],
                [0.25 / sqrt_2, 0.25 / sqrt_2],
                [0.75 / sqrt_2, 0.75 / sqrt_2],
            ]]
        ])

        expected_offset_points = torch.tensor([[[[1.0, 2.0]]], [[[11.0, 12.0]]]])
        expected_scale_distances = torch.tensor([[[[4.0 * sqrt_2]]], [[[4.0 * sqrt_2]]]])

        self.assertTensorClose(normalized_points, expected_normalized_points)
        self.assertTensorClose(offset_points, expected_offset_points)
        self.assertTensorClose(scale_distances, expected_scale_distances)

    def test_profile_normalize(self):
        profile = keypoint_profiles.KeypointProfile2D(
            name='Dummy',
            keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                            ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                            ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
            offset_keypoint_names=['A', 'B'],
            scale_keypoint_name_pairs=[
                (['A', 'B'], ['B']), (['A'], ['B', 'C'])],
            segment_name_pairs=[],
            scale_distance_reduction_fn=torch.sum,
            scale_unit=1.0)
        points = torch.tensor([[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
                  [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]])
        normalized_points, offset_points, scale_distances = profile.normalize(
            points)

        sqrt_2 = 1.414213562
        self.assertTensorClose(normalized_points, torch.tensor([
            [[
                [-0.25 / sqrt_2, -0.25 / sqrt_2],
                [0.25 / sqrt_2, 0.25 / sqrt_2],
                [0.75 / sqrt_2, 0.75 / sqrt_2],
            ]],
            [[
                [-0.25 / sqrt_2, -0.25 / sqrt_2],
                [0.25 / sqrt_2, 0.25 / sqrt_2],
                [0.75 / sqrt_2, 0.75 / sqrt_2],
            ]],
        ]))
        self.assertTensorClose(offset_points, torch.tensor([[[[1.0, 2.0]]], [[[11.0, 12.0]]]]))
        self.assertTensorClose(scale_distances,
                            torch.tensor([[[[4.0 * sqrt_2]]], [[[4.0 * sqrt_2]]]]))

    def test_preprocess_keypoints_3d(self):
        profile = keypoint_profiles.KeypointProfile3D(
            name='Dummy',
            keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                            ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                            ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
            offset_keypoint_names=['A'],
            scale_keypoint_name_pairs=[(['A'], ['B'])],
            segment_name_pairs=[])
        keypoints_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        preprocessed_keypoints_3d, side_outputs = pose_pairs_dataset.preprocess_keypoints_3d(keypoints_3d, profile)

        sqrt_3 = 1.73205080757
        self.assertTensorClose(
            preprocessed_keypoints_3d,
            torch.tensor([[0.0, 0.0, 0.0], [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
             [2.0 / sqrt_3, 2.0 / sqrt_3, 2.0 / sqrt_3]]))
        self.assertCountEqual(
            side_outputs,
            ['offset_points_3d', 'scale_distances_3d', 'preprocessed_keypoints_3d'])
        self.assertTensorClose(
            side_outputs['offset_points_3d'], torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertTensorClose(
            side_outputs['scale_distances_3d'], torch.tensor([[3.0 * sqrt_3]]))
        self.assertTensorClose(
            side_outputs['preprocessed_keypoints_3d'],
            torch.tensor([[0.0, 0.0, 0.0], [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
             [2.0 / sqrt_3, 2.0 / sqrt_3, 2.0 / sqrt_3]]))


    def test_centralize_masked_points(self):
        # Shape = [2, 4, 2].
        points = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                               [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]])
        # Shape = [2, 4].
        point_masks = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

        # Shape = [2, 4, 2].
        centralized_points = keypoint_utils.centralize_masked_points(points, point_masks)

        expected_result = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [3.0, 4.0]],
                                        [[9.0, 10.0], [11.0, 12.0], [10.0, 11.0], [10.0, 11.0]]])
        self.assertTensorClose(centralized_points, expected_result)

    def test_select_keypoints_by_name(self):
        input_keypoints = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [14.0, 14.0, 14.0],
            [15.0, 15.0, 15.0],
            [16.0, 16.0, 16.0],
            [17.0, 17.0, 17.0],
        ])
        keypoint_profile_3d = (
            keypoint_profiles.create_keypoint_profile('EXTRACTED_3DH36M18'))
        keypoint_profile_2d = (
            keypoint_profiles.create_keypoint_profile('LEGACY_2DCOCO13'))
        output_keypoints, _ = keypoint_utils.select_keypoints_by_name(
            input_keypoints,
            input_keypoint_names=keypoint_profile_3d.keypoint_names,
            output_keypoint_names=(
                keypoint_profile_2d.compatible_keypoint_name_dict['EXTRACTED_3DH36M18']
            ))
        self.assertTensorClose(output_keypoints, 
                            torch.tensor([
            [10.0, 10.0, 10.0], # "Head"
            [11.0, 11.0, 11.0], # "LS"
            [14.0, 14.0, 14.0], # "RS"
            [12.0, 12.0, 12.0], # "LE"
            [15.0, 15.0, 15.0], # "RE"
            [13.0, 13.0, 13.0], # "LW"
            [16.0, 16.0, 16.0], # "RW"
            [4.0, 4.0, 4.0],    # "LH"
            [1.0, 1.0, 1.0],    # "RH"
            [5.0, 5.0, 5.0],    # "LK"
            [2.0, 2.0, 2.0],    # "RK"
            [6.0, 6.0, 6.0],    # "LA"
            [3.0, 3.0, 3.0],    # "RA"
            ])
            )

    def test_create_rotation_matrices_3d(self):
        # Shape = [3, 2].
        azimuths = torch.tensor([[0.0, math.pi / 2.0], [math.pi / 2.0, 0.0],
                                [0.0, math.pi / 2.0]])
        elevations = torch.tensor([[0.0, -math.pi / 2.0], [-math.pi / 2.0, 0.0],
                                  [0.0, -math.pi / 2.0]])
        rolls = torch.tensor([[0.0, math.pi], [math.pi, 0.0], [0.0, math.pi]])
        self.assertTensorClose(
            keypoint_utils.create_rotation_matrices_3d(
                azimuths, elevations, rolls),
            torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
              [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
             [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
              [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
             [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
              [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]]
            )
            )

    def test_rotate_by_azimuth_3d(self):
        keypoints_3d = torch.tensor([[
                                        [2.0, 1.0, 3.0],
                                        [1.0, 2.0, -3.0],
                                      ],
                                      [
                                        [3.0, 2.0, -1.0],
                                        [1.0, 2.0, -3.0]
                                      ]])
        keypoints_3d = keypoint_utils.randomly_rotate_3d(
            keypoints_3d,
            azimuth_range=(math.pi / 2.0, math.pi / 2.0),
            elevation_range=(0.0, 0.0),
            roll_range=(0.0, 0.0))
        self.assertTensorClose(keypoints_3d, torch.tensor([[
                                                                [1.0, -2.0, 3.0],
                                                                [2.0, -1.0, -3.0],
                                                            ],
                                                            [
                                                                [2.0, -3.0, -1.0],
                                                                [2.0, -1.0, -3.0]
                                                            ]
                                                            ]))

    def test_rotate_by_elevation_3d(self):
        keypoints_3d = torch.tensor([[[2.0, 1.0, 3.0]]])
        keypoints_3d = keypoint_utils.randomly_rotate_3d(
            keypoints_3d,
            azimuth_range=(0.0, 0.0),
            elevation_range=(math.pi / 2.0, math.pi / 2.0),
            roll_range=(0.0, 0.0))
        self.assertTensorClose(keypoints_3d, torch.tensor([[[2.0, 3.0, -1.0]]]))

    def test_rotate_by_roll_3d(self):
        keypoints_3d = torch.tensor([[[2.0, 1.0, 3.0]]])
        keypoints_3d = keypoint_utils.randomly_rotate_3d(
            keypoints_3d,
            azimuth_range=(0.0, 0.0),
            elevation_range=(0.0, 0.0),
            roll_range=(math.pi / 2.0, math.pi / 2.0))
        self.assertTensorClose(keypoints_3d, torch.tensor([[[-3.0, 1.0, 2.0]]]))

    def test_full_rotation_3d(self):
        keypoints_3d = torch.tensor([[[2.0, 1.0, 3.0]]])
        keypoints_3d = keypoint_utils.randomly_rotate_3d(
            keypoints_3d,
            azimuth_range=(math.pi / 2.0, math.pi / 2.0),
            elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
            roll_range=(math.pi, math.pi))
        self.assertTensorClose(keypoints_3d, torch.tensor([[[3.0, 2.0, 1.0]]]))

    def test_randomly_rotate_and_project_3d_to_2d(self):
        keypoints_3d = torch.tensor([
                                    [[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]],
                                    [[8.0, 7.0, 9.0], [11.0, 10.0, 12.0]],
                                    [[14.0, 13.0, 15.0], [17.0, 16.0, 18.0]]
                                    ])
        keypoints_2d = keypoint_utils.randomly_rotate_and_project_3d_to_2d(
            keypoints_3d,
            azimuth_range=(math.pi / 2.0, math.pi / 2.0),
            elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
            roll_range=(math.pi, math.pi),
            normalized_camera_depth_range=(2.0, 2.0))
        self.assertTensorClose(
            keypoints_2d,torch.tensor(
            [[[-1.0 / 4.0, -3.0 / 4.0], [-4.0 / 7.0, -6.0 / 7.0]],
             [[-7.0 / 10.0, -9.0 / 10.0], [-10.0 / 13.0, -12.0 / 13.0]],
             [[-13.0 / 16.0, -15.0 / 16.0], [-16.0 / 19.0, -18.0 / 19.0]]])
        )

    def test_randomly_rotate_and_project_3d_to_2d_case2(self):
        keypoints_3d = torch.tensor([
            [2.0, 1.0, 3.0],  # HEAD.
            [2.01, 1.01, 3.01],  # NECK.
            [2.02, 1.02, 3.02],  # LEFT_SHOULDER.
            [2.03, 1.03, 3.03],  # RIGHT_SHOULDER.
            [2.04, 1.04, 3.04],  # LEFT_ELBOW.
            [2.05, 1.05, 3.05],  # RIGHT_ELBOW.
            [2.06, 1.06, 3.06],  # LEFT_WRIST.
            [2.07, 1.07, 3.07],  # RIGHT_WRIST.
            [2.08, 1.08, 3.08],  # SPINE.
            [2.09, 1.09, 3.09],  # PELVIS.
            [2.10, 1.10, 3.10],  # LEFT_HIP.
            [2.11, 1.11, 3.11],  # RIGHT_HIP.
            [2.12, 1.12, 3.12],  # LEFT_KNEE.
            [2.13, 1.13, 3.13],  # RIGHT_KNEE.
            [2.14, 1.14, 3.14],  # LEFT_ANKLE.
            [2.15, 1.15, 3.15],  # RIGHT_ANKLE.
        ]
        )
        keypoints_2d = keypoint_utils.randomly_rotate_and_project_3d_to_2d(
            keypoints_3d,
            azimuth_range=(math.pi / 2.0, math.pi / 2.0),
            elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
            roll_range=(math.pi, math.pi),
            normalized_camera_depth_range=(2.0, 2.0))
        self.assertTensorClose(
            keypoints_2d,torch.tensor(
                [[-0.25000006, -0.7500002 ],
                [-0.2518704,  -0.75062364],
                [-0.2537314,  -0.75124395],
                [-0.25558317, -0.75186116],
                [-0.25742579, -0.7524754 ],
                [-0.2592593,  -0.7530866 ],
                [-0.26108378, -0.7536947 ],
                [-0.26289934, -0.7542999 ],
                [-0.26470596, -0.7549021 ],
                [-0.26650375, -0.7555014 ],
                [-0.26829275, -0.7560977 ],
                [-0.27007306, -0.7566911 ],
                [-0.2718447,  -0.7572817 ],
                [-0.2736078,  -0.7578694 ],
                [-0.27536237, -0.7584543 ],
                [-0.2771085,  -0.7590363 ]]
             )
        )

    def test_randomly_project_and_select_keypoints(self):
        keypoints_3d = torch.tensor([
            [2.0, 1.0, 3.0],  # Pelvis.
            [2.01, 1.01, 3.01],  # R_Hip.
            [2.02, 1.02, 3.02],  # R_Knee
            [2.03, 1.03, 3.03],  # R_Ankle
            [2.04, 1.04, 3.04],  # L_Hip
            [2.05, 1.05, 3.05],  # L_Knee
            [2.06, 1.06, 3.06],  # L_Ankle
            [2.07, 1.07, 3.07],  # Torso
            [2.08, 1.08, 3.08],  # Neck
            [2.09, 1.09, 3.09],  # Nose
            [2.10, 1.10, 3.10],  # Head
            [2.11, 1.11, 3.11],  # L_Shoulder
            [2.12, 1.12, 3.12],  # L_Elbow
            [2.13, 1.13, 3.13],  # L_Wrist
            [2.14, 1.14, 3.14],  # R_Shoulder
            [2.15, 1.15, 3.15],  # R_Elbow
            [2.16, 1.16, 3.16],  # R_Wrist
            [2.17, 1.17, 3.17],  # Thorax
        ])
        keypoint_profile_3d = (
            keypoint_profiles.create_keypoint_profile('EXTRACTED_3DH36M18'))
        keypoint_profile_2d = (
            keypoint_profiles.create_keypoint_profile('LEGACY_2DCOCO13'))
        keypoints_2d, _ = keypoint_utils.randomly_project_and_select_keypoints(
            keypoints_3d,
            keypoint_profile_3d=keypoint_profile_3d,
            output_keypoint_names=(
                keypoint_profile_2d.compatible_keypoint_name_dict['EXTRACTED_3DH36M18']),
            azimuth_range=(math.pi / 2.0, math.pi / 2.0),
            elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
            roll_range=(math.pi, math.pi),
            normalized_camera_depth_range=(2.0, 2.0),
            normalize_before_projection=False)
        self.assertTensorClose(
            keypoints_2d,
            torch.tensor(
            [
                [-1.10 / 4.10, -3.10 / 4.10],  # NOSE_TIP
                [-1.11 / 4.11, -3.11 / 4.11],  # LEFT_SHOULDER.
                [-1.14 / 4.14, -3.14 / 4.14],  # RIGHT_SHOULDER.
                [-1.12 / 4.12, -3.12 / 4.12],  # LEFT_ELBOW.
                [-1.15 / 4.15, -3.15 / 4.15],  # RIGHT_ELBOW.
                [-1.13 / 4.13, -3.13 / 4.13],  # LEFT_WRIST.
                [-1.16 / 4.16, -3.16 / 4.16],  # RIGHT_WRIST.
                [-1.04 / 4.04, -3.04 / 4.04],  # LEFT_HIP.
                [-1.01 / 4.01, -3.01 / 4.01],  # RIGHT_HIP.
                [-1.05 / 4.05, -3.05 / 4.05],  # LEFT_KNEE.
                [-1.02 / 4.02, -3.02 / 4.02],  # RIGHT_KNEE.
                [-1.06 / 4.06, -3.06 / 4.06],  # LEFT_ANKLE.
                [-1.03 / 4.03, -3.03 / 4.03],  # RIGHT_ANKLE.
            ])
        )

    def test_transfer_keypoint_masks_case_1(self):
        # Shape = [2, 13].
        input_keypoint_masks = torch.tensor([
            [
                1.0,  # HEAD
                1.0,  # LEFT_SHOULDER
                1.0,  # RIGHT_SHOULDER
                0.0,  # LEFT_ELBOW
                1.0,  # RIGHT_ELBOW
                1.0,  # LEFT_WRIST
                0.0,  # RIGHT_WRIST
                1.0,  # LEFT_HIP
                0.0,  # RIGHT_HIP
                1.0,  # LEFT_KNEE
                1.0,  # RIGHT_KNEE
                0.0,  # LEFT_ANKLE
                0.0,  # RIGHT_ANKLE
            ],
            [
                0.0,  # HEAD
                0.0,  # LEFT_SHOULDER
                0.0,  # RIGHT_SHOULDER
                1.0,  # LEFT_ELBOW
                0.0,  # RIGHT_ELBOW
                0.0,  # LEFT_WRIST
                1.0,  # RIGHT_WRIST
                0.0,  # LEFT_HIP
                1.0,  # RIGHT_HIP
                0.0,  # LEFT_KNEE
                0.0,  # RIGHT_KNEE
                1.0,  # LEFT_ANKLE
                1.0,  # RIGHT_ANKLE
            ]
        ])
        input_keypoint_profile = keypoint_profiles.create_keypoint_profile(
            'LEGACY_2DCOCO13')
        output_keypoint_profile = keypoint_profiles.create_keypoint_profile(
            'EXTRACTED_3DH36M18')
        # Shape = [2, 18].
        output_keypoint_masks = keypoint_utils.transfer_keypoint_masks(
            input_keypoint_masks, input_keypoint_profile, output_keypoint_profile)
        """

        """
        self.assertTensorClose(
            output_keypoint_masks,
            torch.tensor(
            [
                [
                    0.0,  # Pelvis.
                    0.0,  # R_Hip.
                    1.0,  # R_Knee
                    0.0,  # R_Ankle
                    1.0,  # L_Hip
                    1.0,  # L_Knee
                    0.0,  # L_Ankle
                    0.0,  # Torso
                    1.0,  # Neck
                    0.0,  # Nose
                    1.0,  # Head
                    1.0,  # L_Shoulder
                    0.0,  # L_Elbow
                    1.0,  # L_Wrist
                    1.0,  # R_Shoulder
                    1.0,  # R_Elbow
                    0.0,  # R_Wrist
                    0.0,  # Thorax                    
                ],
                [
                    0.0,  # Pelvis.
                    1.0,  # R_Hip.
                    0.0,  # R_Knee
                    1.0,  # R_Ankle
                    0.0,  # L_Hip
                    0.0,  # L_Knee
                    1.0,  # L_Ankle
                    0.0,  # Torso
                    0.0,  # Neck
                    0.0,  # Nose
                    0.0,  # Head
                    0.0,  # L_Shoulder
                    1.0,  # L_Elbow
                    0.0,  # L_Wrist
                    0.0,  # R_Shoulder
                    0.0,  # R_Elbow
                    1.0,  # R_Wrist
                    0.0,  # Thorax                    
                ]
            ])
        )

    def test_transfer_keypoint_masks_case_2(self):
        # Shape = [2, 16].
        input_keypoint_masks = torch.tensor([
            [
                0.0,  # Pelvis.
                0.0,  # R_Hip.
                1.0,  # R_Knee
                0.0,  # R_Ankle
                1.0,  # L_Hip
                1.0,  # L_Knee
                0.0,  # L_Ankle
                0.0,  # Torso
                1.0,  # Neck
                0.0,  # Nose
                1.0,  # Head
                0.0,  # L_Shoulder
                0.0,  # L_Elbow
                1.0,  # L_Wrist
                1.0,  # R_Shoulder
                1.0,  # R_Elbow
                0.0,  # R_Wrist
                0.0,  # Thorax   
            ],
            [
                0.0,  # Pelvis.
                1.0,  # R_Hip.
                0.0,  # R_Knee
                1.0,  # R_Ankle
                0.0,  # L_Hip
                0.0,  # L_Knee
                1.0,  # L_Ankle
                0.0,  # Torso
                0.0,  # Neck
                0.0,  # Nose
                0.0,  # Head
                0.0,  # L_Shoulder
                1.0,  # L_Elbow
                0.0,  # L_Wrist
                0.0,  # R_Shoulder
                0.0,  # R_Elbow
                1.0,  # R_Wrist
                0.0,  # Thorax   
            ]
        ])
        input_keypoint_profile = keypoint_profiles.create_keypoint_profile(
            'EXTRACTED_3DH36M18')
        output_keypoint_profile = keypoint_profiles.create_keypoint_profile(
            'LEGACY_2DCOCO13')
        # Shape = [2, 13].
        output_keypoint_masks = keypoint_utils.transfer_keypoint_masks(
            input_keypoint_masks, input_keypoint_profile, output_keypoint_profile)
        self.assertTensorClose(
            output_keypoint_masks, torch.tensor(
            [
                [
                    1.0,  # NOSE_TIP
                    0.0,  # LEFT_SHOULDER
                    1.0,  # RIGHT_SHOULDER
                    0.0,  # LEFT_ELBOW
                    1.0,  # RIGHT_ELBOW
                    1.0,  # LEFT_WRIST
                    0.0,  # RIGHT_WRIST
                    1.0,  # LEFT_HIP
                    0.0,  # RIGHT_HIP
                    1.0,  # LEFT_KNEE
                    1.0,  # RIGHT_KNEE
                    0.0,  # LEFT_ANKLE
                    0.0,  # RIGHT_ANKLE
                ],
                [
                    0.0,  # NOSE_TIP
                    0.0,  # LEFT_SHOULDER
                    0.0,  # RIGHT_SHOULDER
                    1.0,  # LEFT_ELBOW
                    0.0,  # RIGHT_ELBOW
                    0.0,  # LEFT_WRIST
                    1.0,  # RIGHT_WRIST
                    0.0,  # LEFT_HIP
                    1.0,  # RIGHT_HIP
                    0.0,  # LEFT_KNEE
                    0.0,  # RIGHT_KNEE
                    1.0,  # LEFT_ANKLE
                    1.0,  # RIGHT_ANKLE
                ]
            ])
        )
    
# Run the tests
if __name__ == '__main__':
    unittest.main()
