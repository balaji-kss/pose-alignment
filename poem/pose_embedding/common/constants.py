# Embedding distance types.
# Distance computed using embedding centers.
DISTANCE_TYPE_CENTER = 'CENTER'
# Distance computed using embedding samples.
DISTANCE_TYPE_SAMPLE = 'SAMPLE'
# Distance computed using both embedding centers and samples.
DISTANCE_TYPE_CENTER_AND_SAMPLE = 'CENTER_AND_SAMPLE'
# Supported distance types.
SUPPORTED_DISTANCE_TYPES = [
    DISTANCE_TYPE_CENTER,
    DISTANCE_TYPE_SAMPLE,
    DISTANCE_TYPE_CENTER_AND_SAMPLE,
]

# Embedding distance pair types.
# Reduces distances between all pairs between two lists of samples.
DISTANCE_PAIR_TYPE_ALL_PAIRS = 'ALL_PAIRS'
# Reduces distances only between corrresponding pairs between two lists of
# samples.
DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS = 'CORRESPONDING_PAIRS'
# Supported distance pair types.
SUPPORTED_DISTANCE_PAIR_TYPES = [
    DISTANCE_PAIR_TYPE_ALL_PAIRS,
    DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS,
]

# Embedding distance kernels.
# Squared L2 distance.
DISTANCE_KERNEL_SQUARED_L2 = 'SQUARED_L2'
# L2-based sigmoid matching probability.
DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB = 'L2_SIGMOID_MATCHING_PROB'
# Squared L2-based sigmoid matching probability.
DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB = (
    'SQUARED_L2_SIGMOID_MATCHING_PROB')
# Expected likelihood.
DISTANCE_KERNEL_EXPECTED_LIKELIHOOD = 'EXPECTED_LIKELIHOOD'
# Supported distance kernels.
SUPPORTED_DISTANCE_KERNELS = [
    DISTANCE_KERNEL_SQUARED_L2,
    DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
    DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB,
    DISTANCE_KERNEL_EXPECTED_LIKELIHOOD,
]


# Embedding distance reductions.
# Mean of all distances.
DISTANCE_REDUCTION_MEAN = 'MEAN'
# Mean of distances not larger than the median of all distances.
DISTANCE_REDUCTION_LOWER_HALF_MEAN = 'LOWER_HALF_MEAN'
# Negative logarithm of the mean of all distances.
DISTANCE_REDUCTION_NEG_LOG_MEAN = 'NEG_LOG_MEAN'
# Negative logarithm of mean of distances no larger than the distance median.
DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN = 'LOWER_HALF_NEG_LOG_MEAN'
# One minus the mean of all distances.
DISTANCE_REDUCTION_ONE_MINUS_MEAN = 'ONE_MINUS_MEAN'
# Supported embedding distance reductions.
SUPPORTED_PAIRWISE_DISTANCE_REDUCTIONS = [
    DISTANCE_REDUCTION_MEAN,
    DISTANCE_REDUCTION_LOWER_HALF_MEAN,
    DISTANCE_REDUCTION_NEG_LOG_MEAN,
    DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN,
    DISTANCE_REDUCTION_ONE_MINUS_MEAN,
]
SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS = [DISTANCE_REDUCTION_MEAN]

# 3D keypoint distance measurement type.
# Normalized/Procrustes-aligned MPJPE.
KEYPOINT_DISTANCE_TYPE_MPJPE = 'MPJPE'
# Supported 3D keypoint distance measurement type.
SUPPORTED_KEYPOINT_DISTANCE_TYPES = [KEYPOINT_DISTANCE_TYPE_MPJPE]

# Base model types.
# Simple Baseline architecutre: Martinez, et al. A simple yet effective baseline
# for 3d human pose estimation. ICCV 2017.
BASE_MODEL_TYPE_SIMPLE = 'SIMPLE'
# Temporal Simple Baseline.
BASE_MODEL_TYPE_TEMPORAL_SIMPLE = 'TEMPORAL_SIMPLE'
# Temporal Simple Baseline late fusion.
BASE_MODEL_TYPE_TEMPORAL_SIMPLE_LATE_FUSE = 'TEMPORAL_SIMPLE_LATE_FUSE'
# Supported base model types.
SUPPORTED_BASE_MODEL_TYPES = [
    BASE_MODEL_TYPE_SIMPLE,
    BASE_MODEL_TYPE_TEMPORAL_SIMPLE,
    BASE_MODEL_TYPE_TEMPORAL_SIMPLE_LATE_FUSE,
]

KEY_KEYPOINTS_2D = 'keypoints_2d'
KEY_KEYPOINT_SCORES_2D = 'keypoint_scores_2d'
KEY_KEYPOINT_MASKS_2D = 'keypoint_masks_2d'
KEY_PREPROCESSED_KEYPOINTS_2D = 'preprocessed_keypoints_2d'
KEY_PREPROCESSED_KEYPOINT_MASKS_2D = 'preprocessed_keypoint_masks_2d'
KEY_OFFSET_POINTS_2D = 'offset_points_2d'
KEY_SCALE_DISTANCES_2D = 'scale_distances_2d'
KEY_KEYPOINTS_3D = 'keypoints_3d'
KEY_PREPROCESSED_KEYPOINTS_3D = 'preprocessed_keypoints_3d'
KEY_OFFSET_POINTS_3D = 'offset_points_3d'
KEY_SCALE_DISTANCES_3D = 'scale_distances_3d'
KEY_EMBEDDING_MEANS = 'unnormalized_embeddings'
KEY_EMBEDDING_STDDEVS = 'embedding_stddevs'
KEY_EMBEDDING_SAMPLES = 'unnormalized_embedding_samples'

# Embedding types.
# Point embedding.
EMBEDDING_TYPE_POINT = 'POINT'
# Gaussian embedding with diagonal covariance matrix.
EMBEDDING_TYPE_GAUSSIAN = 'GAUSSIAN'
# Gaussian embedding with scalar variance.
EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR = 'GAUSSIAN_SCALAR_VAR'
# Supported embedding types.
SUPPORTED_EMBEDDING_TYPES = [
    EMBEDDING_TYPE_POINT,
    EMBEDDING_TYPE_GAUSSIAN,
    EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR,
]


sigmoid_raw_a_initial= -0.65
sigmoid_b_initial = -0.5
sigmoid_a_max = -1.0
min_negative_keypoint_mpjpe = 0.1
triplet_distance_kernel = DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB
triplet_distance_type = DISTANCE_TYPE_SAMPLE
triplet_componentwise_reduction = DISTANCE_REDUCTION_MEAN
triplet_pairwise_reduction = DISTANCE_REDUCTION_NEG_LOG_MEAN

positive_pairwise_distance_kernel = triplet_distance_kernel
positive_pairwise_pairwise_reduction = triplet_pairwise_reduction
positive_pairwise_componentwise_reduction = triplet_componentwise_reduction
triplet_loss_margin = 0.69314718056
use_semi_hard_triplet_negatives = True

embedding_size = 16
num_embedding_samples = 20
num_embedding_components = 1
dropout_rate = 0.3

kl_regularization_loss_weight = 0.001
kl_regularization_prior_stddev = 1.0
positive_pairwise_loss_weight = 0.005

min_stddev = 0.1
max_squared_mahalanobis_distance = 100.0

learning_rate_schedule = "" # ['', 'EXP_DECAY', 'LINEAR_DECAY'],
learning_rate = 0.02    # initial learning rate
num_steps = 5_000_000
decay_steps=num_steps    # 5,000,000
num_warmup_steps = None

optimizer_name = 'ADAGRAD' # ['ADAGRAD', 'ADAM', 'ADAMW']

draw_anchor_positive_number = 8

threshold_matching_prob_dist = 0.3