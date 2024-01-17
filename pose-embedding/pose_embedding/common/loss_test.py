import torch
import numpy as np
import unittest
import math
import functools
from pose_embedding.common import constants
from pose_embedding.common import loss_utils



class LossTestOperations(unittest.TestCase):

    def _assert_dict_equal_or_almost_equal(self, result, expected_result, float_equal_places=4):
        self.assertCountEqual(result.keys(), expected_result.keys())
        for key, ev in expected_result.items():
            value = result[key].cpu().numpy() if torch.is_tensor(result[key]) else result[key]
            
            if isinstance(ev, int):
                self.assertEqual(value, ev, msg='Key = `%s`.' % key)
            elif isinstance(ev, float):
                self.assertAlmostEqual(value, ev, places=float_equal_places, msg='Key = `%s`.' % key)
            elif isinstance(ev, (list, tuple)):
                if ev:
                    if isinstance(ev[0], int):
                        self.assertListEqual(list(value), ev, msg='Key = `%s`.' % key)
                    elif isinstance(ev[0], float):
                        for v, e in zip(value, ev):
                            self.assertAlmostEqual(v, e, places=float_equal_places, msg='Key = `%s`.' % key)
                    else:
                        raise ValueError('Unsupported expected value type for key `%s`: list/tuple of %s.' % (key, type(ev[0])))
                else:
                    self.assertEqual(list(value), ev, msg='Key = `%s`.' % key)
            else:
                raise ValueError('Unsupported expected value type for key `%s`: %s.' % (key, type(ev)))

    def assertTensorClose(self, a, b, atol=1e-5, rtol=1e-4):
        """
        Custom assert function to check if two tensors are close
        """
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol),
                        f"Tensors are not close: \n{a} \n{b}")

    def test_compute_lower_percentile_means(self):
        # Shape = [2, 3, 3].
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                         [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0],
                          [17.0, 18.0, 19.0]]])
        lower_half = loss_utils.compute_lower_percentile_means(
            x, axis=[-2, -1])
        self.assertTensorClose(lower_half, torch.tensor([3.0, 13.0]))

    def test_get_sigmoid_parameters(self):
        raw_a, a, b = loss_utils.get_sigmoid_parameters(
            raw_a_initial_value=1.0,
            b_initial_value=2.0,
            a_range=(-0.5, 1.2),
            b_range=(3.0, 5.0)
        )

        # Note: In PyTorch, there's no need for a session. You can directly evaluate tensors.
        raw_a_result = raw_a.item()  # Convert single-value tensor to Python scalar
        a_result = a.item()
        b_result = b.item()

        # Use the assertAlmostEqual to check if the values are close enough
        self.assertAlmostEqual(raw_a_result, 1.0, places=5)  # places determines the number of decimal places
        self.assertAlmostEqual(a_result, 1.2, places=5)
        self.assertAlmostEqual(b_result, 3.0, places=5)

    def test_create_sample_distance_fn_case_1(self):
        def pairwise_reduction_fn(x):
            min_over_last_dim, _ = torch.min(x, dim=-1)
            min_over_second_last_dim, _ = torch.min(min_over_last_dim, dim=-1)
            return min_over_second_last_dim
        # Shape = [2, 3, 2].
        lhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
        # Shape = [2, 4, 2].
        rhs = torch.tensor([[[16.0, 15.0], [14.0, 13.0], [12.0, 11.0], [10.0, 9.0]],
                           [[8.0, 7.0], [6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
        distances = loss_utils.create_sample_distance_fn(
            pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
            distance_kernel=constants.DISTANCE_KERNEL_SQUARED_L2,
            pairwise_reduction=pairwise_reduction_fn, #functools.partial(
                # torch.min, dim=(-2, -1), keepdim=True),
            componentwise_reduction=lambda x: x)(lhs, rhs)

        self.assertTensorClose(distances, torch.tensor([34.0, 2.0]))

    def test_create_sample_distance_fn_case_2(self):
        # Shape = [1, 2, 2].
        lhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        # Shape = [1, 3, 2].
        rhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        distances = loss_utils.create_sample_distance_fn(
            pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
            distance_kernel=constants.DISTANCE_KERNEL_SQUARED_L2,
            pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
            componentwise_reduction=lambda x: x)(lhs, rhs)

        self.assertTensorClose(distances, torch.tensor([56.0 / 6.0]))

    def test_compute_negative_indicator_matrix(self):
        anchors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        matches = torch.tensor([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
        indicators = loss_utils.compute_negative_indicator_matrix(
            anchors,
            matches,
            distance_fn=lambda t1, t2: (t1-t2)**2,
            min_negative_distance=5.0)

        self.assertTensorClose(indicators,
                            torch.tensor([[[True, True], [True, False], [False, False]],
                             [[True, False], [False, False], [False, True]]]))

    def test_compute_hard_negative_distances(self):
        anchor_match_distance_matrix = torch.tensor([
            [1.0, 3.0, 2.0, 0.0],
            [6.0, 5.0, 4.0, 0.0],
            [7.0, 8.0, 9.0, 10.0],
        ])
        negative_indicator_matrix = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
        ])
        hard_negative_distances, hard_negative_mining_distances = (
            loss_utils.compute_hard_negative_distances(anchor_match_distance_matrix,
                                                       negative_indicator_matrix))

        self.assertTrue(torch.allclose(hard_negative_distances, torch.tensor([1.0, 5.0, float('inf')])))
        self.assertTrue(torch.allclose(hard_negative_mining_distances, torch.tensor([1.0, 5.0, float('inf')])))

    def test_compute_semi_hard_negative_triplet_loss(self):
        anchor_positive_distances = torch.tensor([0.0, 8.0, 2.0])
        anchor_match_distance_matrix = torch.tensor([
            [2.0, 8.0, 32.0, 72.0],
            [2.0, 0.0, 8.0, 32.0],
            [18.0, 8.0, 0.0, 8.0],
        ])
        anchor_match_negative_indicator_matrix = torch.tensor([
            [True, True, True, True],
            [True, True, False, True],
            [True, False, True, False],
        ], dtype=torch.bool)

        (loss, num_active_triplets, anchor_negative_distances, mining_loss,
         num_active_mining_triplets, anchor_negative_mining_distances) = (
             loss_utils.compute_hard_negative_triplet_loss(
                 anchor_positive_distances,
                 anchor_match_distance_matrix,
                 anchor_match_negative_indicator_matrix,
                 margin=20.0,
                 use_semi_hard=True))

        self.assertAlmostEqual(loss.item(), 22.0 / 3.0, places=4)
        self.assertEqual(num_active_triplets, 2)
        self.assertTensorClose(anchor_negative_distances, torch.tensor([2.0, 32.0, 18.0]))
        self.assertAlmostEqual(mining_loss.item(), 22.0 / 3.0, places=4)
        self.assertEqual(num_active_mining_triplets, 2)
        self.assertTensorClose(anchor_negative_mining_distances, torch.tensor([2.0, 32.0, 18.0]))

    def test_compute_keypoint_triplet_losses(self):
        # Shape = [3, 1, 1, 2].
        anchor_embeddings = torch.tensor([
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ])
        # Shape = [3, 1, 1, 2].
        positive_embeddings = torch.tensor([
            [[[1.0, 2.0]]],
            [[[5.0, 6.0]]],
            [[[6.0, 7.0]]],
        ])
        # Shape = [4, 1, 1, 2].
        match_embeddings = torch.tensor([
            [[[2.0, 3.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
            [[[7.0, 8.0]]],
        ])
        # Shape = [3, 1].
        anchor_keypoints = torch.tensor([[1], [2], [3]])
        # Shape = [4, 1].
        match_keypoints = torch.tensor([[1], [2], [3], [4]])

        def mock_keypoint_distance_fn(unused_lhs, unused_rhs):
            # Shape = [3, 4].
            return torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 1.0, 0.0]])

        loss, summaries = loss_utils.compute_keypoint_triplet_losses(
            anchor_embeddings,
            positive_embeddings,
            match_embeddings,
            anchor_keypoints,
            match_keypoints,
            margin=20.0,
            min_negative_keypoint_distance=0.5,
            use_semi_hard=True,
            exclude_inactive_triplet_loss=True,
            keypoint_distance_fn=mock_keypoint_distance_fn)

        self.assertAlmostEqual(loss, 11.0)

        expected_summaries = {
            'triplet_loss/Margin': 20.0,
            'triplet_loss/Anchor/Positive/Distance/Mean': 10.0 / 3,
            'triplet_loss/Anchor/Positive/Distance/Median': 2.0,
            'triplet_loss/Anchor/HardNegative/Distance/Mean': 2.0 / 3,
            'triplet_loss/Anchor/HardNegative/Distance/Median': 0.0,
            'triplet_loss/Anchor/SemiHardNegative/Distance/Mean': 52.0 / 3,
            'triplet_loss/Anchor/SemiHardNegative/Distance/Median': 18.0,
            'triplet_loss/HardNegative/Loss/All': 68.0 / 3,
            'triplet_loss/HardNegative/Loss/Active': 68.0 / 3,
            'triplet_loss/HardNegative/ActiveTripletNum': 3,
            'triplet_loss/HardNegative/ActiveTripletRatio': 1.0,
            'triplet_loss/SemiHardNegative/Loss/All': 22.0 / 3,
            'triplet_loss/SemiHardNegative/Loss/Active': 22.0 / 2,
            'triplet_loss/SemiHardNegative/ActiveTripletNum': 2,
            'triplet_loss/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
            'triplet_mining/Anchor/Positive/Distance/Mean': 10.0 / 3,
            'triplet_mining/Anchor/Positive/Distance/Median': 2.0,
            'triplet_mining/Anchor/HardNegative/Distance/Mean': 2.0 / 3,
            'triplet_mining/Anchor/HardNegative/Distance/Median': 0.0,
            'triplet_mining/Anchor/SemiHardNegative/Distance/Mean': 52.0 / 3,
            'triplet_mining/Anchor/SemiHardNegative/Distance/Median': 18.0,
            'triplet_mining/HardNegative/Loss/All': 68.0 / 3,
            'triplet_mining/HardNegative/Loss/Active': 68.0 / 3,
            'triplet_mining/HardNegative/ActiveTripletNum': 3,
            'triplet_mining/HardNegative/ActiveTripletRatio': 1.0,
            'triplet_mining/SemiHardNegative/Loss/All': 22.0 / 3,
            'triplet_mining/SemiHardNegative/Loss/Active': 22.0 / 2,
            'triplet_mining/SemiHardNegative/ActiveTripletNum': 2,
            'triplet_mining/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
        }
        self._assert_dict_equal_or_almost_equal(summaries, expected_summaries)

    def test_compute_kl_regularization_loss(self):
        means = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stddevs = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        weighted_loss, summaries = loss_utils.compute_kl_regularization_loss(
            means, stddevs, loss_weight=3.0)

        self.assertAlmostEqual(weighted_loss.numpy(), 122.131123182, places=4)

        expected_summaries = {
            'regularization_loss/KL/PriorMean/Mean': 0.0,
            'regularization_loss/KL/PriorVar/Mean': 1.0,
            'regularization_loss/KL/Loss/Original': 40.710374394,
            'regularization_loss/KL/Loss/Weighted': 122.131123182,
            'regularization_loss/KL/Loss/Weight': 3.0,
        }
        self._assert_dict_equal_or_almost_equal(summaries, expected_summaries)


    def test_compute_positive_pairwise_loss(self):
        anchor_embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                         [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
        positive_embeddings = torch.tensor([[[12.0, 11.0], [10.0, 9.0], [8.0, 7.0]],
                                           [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
        weighted_loss, summaries = loss_utils.compute_positive_pairwise_loss(
            anchor_embeddings, positive_embeddings, loss_weight=6.0)

        self.assertAlmostEqual(weighted_loss, 572.0)

        expected_summaries = {
            'pairwise_loss/PositivePair/Loss/Original': 95.333333333,
            'pairwise_loss/PositivePair/Loss/Weighted': 572.0,
            'pairwise_loss/PositivePair/Loss/Weight': 6.0,
        }
        self._assert_dict_equal_or_almost_equal(summaries, expected_summaries)

# Run the tests
if __name__ == '__main__':
    unittest.main()
