import numpy as np

def _normalize(vec):

    den = np.linalg.norm(vec)

    if den == 0: return np.array([0.0, 0.0, 0.0])

    return vec / np.linalg.norm(vec)

def get_trunk_vector(joints_3d):

    lshoulder = joints_3d[11]
    rshoulder = joints_3d[14]
    thorax = joints_3d[8]
    spine = joints_3d[7]

    lhip = joints_3d[4]
    rhip = joints_3d[1]
    pelvis = joints_3d[0]

    up_body = np.array([lshoulder, rshoulder, thorax, spine])
    low_body = np.array([lhip, rhip, pelvis])

    avg_up = up_body.mean(axis=0)
    avg_down = low_body.mean(axis=0)

    trunk_vector = _normalize(avg_up - avg_down)
    
    return trunk_vector, [avg_up, avg_down]

def get_trunk_twist_dev(base_joints_3d, cand_joints_3d):

    base_shoulder_vector, _ = get_shoulder_vector(base_joints_3d)
    cand_shoulder_vector, _ = get_shoulder_vector(cand_joints_3d)

    base_hip_vector, _ = get_hip_vector(base_joints_3d)
    cand_hip_vector, _ = get_hip_vector(cand_joints_3d)

    base_trunk_twist_angle = np.arccos(np.dot(base_hip_vector, base_shoulder_vector)) * 180.0 / np.pi
    cand_trunk_twist_angle = np.arccos(np.dot(cand_hip_vector, cand_shoulder_vector)) * 180.0 / np.pi

    # print('cand_trunk_twist_angle ', cand_trunk_twist_angle, ' base_trunk_twist_angle ', base_trunk_twist_angle)
    diff_angle = abs(cand_trunk_twist_angle - base_trunk_twist_angle)

    return np.round(diff_angle, 2) 

def get_trunk_dev(base_joints_3d, cand_joints_3d):

    base_trunk_vector, _ = get_trunk_vector(base_joints_3d)
    cand_trunk_vector, _ = get_trunk_vector(cand_joints_3d)
    ref_vector = np.array([0.0, 0.0, 1])

    base_trunk_angle = np.arccos(np.dot(base_trunk_vector, ref_vector)) * 180.0 / np.pi
    cand_trunk_angle = np.arccos(np.dot(cand_trunk_vector, ref_vector)) * 180.0 / np.pi

    diff_angle = abs(cand_trunk_angle - base_trunk_angle)

    return np.round(diff_angle, 2)

def get_left_arm_vector(joints_3d):

    lshoulder = joints_3d[11]
    lelbow = joints_3d[12]

    larm_vector = _normalize(lelbow - lshoulder)  

    return larm_vector, [lshoulder, lelbow]

def get_left_arm_dev0(base_joints_3d, cand_joints_3d):

    blarm_vector, _ = get_left_arm_vector(base_joints_3d)
    clarm_vector, _ = get_left_arm_vector(cand_joints_3d)

    larm_dev = np.arccos(np.dot(clarm_vector, blarm_vector)) * 180.0 / np.pi

    return np.round(larm_dev, 2)

def get_left_arm_dev(base_joints_3d, cand_joints_3d):

    blarm_vector, _ = get_left_arm_vector(base_joints_3d)
    clarm_vector, _ = get_left_arm_vector(cand_joints_3d)
    
    larm_dev = np.arccos(np.dot(clarm_vector, blarm_vector)) * 180.0 / np.pi

    return np.round(larm_dev, 2)

def get_left_arm_dev0(base_joints_3d, cand_joints_3d):

    blarm_vector, _ = get_left_arm_vector(base_joints_3d)
    clarm_vector, _ = get_left_arm_vector(cand_joints_3d)

    larm_dev = np.arccos(np.dot(clarm_vector, blarm_vector)) * 180.0 / np.pi

    return np.round(larm_dev, 2)

def get_right_arm_vector(joints_3d):

    rshoulder = joints_3d[14]
    relbow = joints_3d[15]

    rarm_vector = _normalize(relbow - rshoulder)  

    return rarm_vector, [rshoulder, relbow]

def get_left_forearm_vector(joints_3d):

    lelbow = joints_3d[12]
    lwrist = joints_3d[13]

    lfarm_vector = _normalize(lwrist - lelbow)  

    return lfarm_vector, [lwrist, lelbow]

def get_right_forearm_vector(joints_3d):

    relbow = joints_3d[15]
    rwrist = joints_3d[16]

    rfarm_vector = _normalize(rwrist - relbow)  

    return rfarm_vector, [rwrist, relbow]

def get_left_farm_dev(base_joints_3d, cand_joints_3d):

    blfarm_vector, _ = get_left_forearm_vector(base_joints_3d)
    clfarm_vector, _ = get_left_forearm_vector(cand_joints_3d)

    blarm_vector, _ = get_left_arm_vector(base_joints_3d)
    clarm_vector, _ = get_left_arm_vector(cand_joints_3d)

    base_angle = np.arccos(np.dot(blfarm_vector, blarm_vector)) * 180.0 / np.pi
    cand_angle = np.arccos(np.dot(clfarm_vector, clarm_vector)) * 180.0 / np.pi

    # print('base_angle ', base_angle, ' cand_angle ', cand_angle)
    diff_angle = abs(cand_angle - base_angle)

    return np.round(diff_angle, 2)

def get_right_farm_dev(base_joints_3d, cand_joints_3d):

    brfarm_vector, _ = get_right_forearm_vector(base_joints_3d)
    crfarm_vector, _ = get_right_forearm_vector(cand_joints_3d)

    brarm_vector, _ = get_right_arm_vector(base_joints_3d)
    crarm_vector, _ = get_right_arm_vector(cand_joints_3d)

    base_angle = np.arccos(np.dot(brfarm_vector, brarm_vector)) * 180.0 / np.pi
    cand_angle = np.arccos(np.dot(crfarm_vector, crarm_vector)) * 180.0 / np.pi

    # print('base_angle ', base_angle, ' cand_angle ', cand_angle)
    diff_angle = abs(cand_angle - base_angle)

    return np.round(diff_angle, 2)

def get_right_arm_dev(base_joints_3d, cand_joints_3d):

    brarm_vector, _ = get_right_arm_vector(base_joints_3d)
    crarm_vector, _ = get_right_arm_vector(cand_joints_3d)

    rarm_dev = np.arccos(np.dot(crarm_vector, brarm_vector)) * 180.0 / np.pi

    return np.round(rarm_dev, 2)

def get_left_thigh_vector(joints_3d):

    lhip = joints_3d[4]
    lknee = joints_3d[5]

    lthigh_vector = _normalize(lknee - lhip)  

    return lthigh_vector, [lknee, lhip]

def get_right_thigh_vector(joints_3d):

    rhip = joints_3d[1]
    rknee = joints_3d[2]

    rthigh_vector = _normalize(rknee - rhip)  

    return rthigh_vector, [rknee, rhip]

def get_left_thigh_dev(base_joints_3d, cand_joints_3d):

    blthigh_vector, _ = get_left_thigh_vector(base_joints_3d)
    clthigh_vector, _ = get_left_thigh_vector(cand_joints_3d)

    lthigh_dev = np.arccos(np.dot(clthigh_vector, blthigh_vector)) * 180.0 / np.pi

    return np.round(lthigh_dev, 2)

def get_right_thigh_dev(base_joints_3d, cand_joints_3d):

    brthigh_vector, _ = get_right_thigh_vector(base_joints_3d)
    crthigh_vector, _ = get_right_thigh_vector(cand_joints_3d)

    rthigh_dev = np.arccos(np.dot(crthigh_vector, brthigh_vector)) * 180.0 / np.pi

    return np.round(rthigh_dev, 2)

def get_left_leg_vector(joints_3d):

    lknee = joints_3d[5]
    lankle = joints_3d[6]

    lleg_vector = _normalize(lankle - lknee)  

    return lleg_vector, [lankle, lknee]

def get_right_leg_vector(joints_3d):

    rknee = joints_3d[2]
    rankle = joints_3d[3]

    rleg_vector = _normalize(rankle - rknee)  

    return rleg_vector, [rankle, rknee]

def get_left_leg_dev(base_joints_3d, cand_joints_3d):

    blleg_vector, _ = get_left_leg_vector(base_joints_3d)
    clleg_vector, _ = get_left_leg_vector(cand_joints_3d)

    blthigh_vector, _ = get_left_thigh_vector(base_joints_3d)
    clthigh_vector, _ = get_left_thigh_vector(cand_joints_3d)

    base_angle = np.arccos(np.dot(blthigh_vector, blleg_vector)) * 180.0 / np.pi
    cand_angle = np.arccos(np.dot(clthigh_vector, clleg_vector)) * 180.0 / np.pi

    # print('base_angle ', base_angle, ' cand_angle ', cand_angle)

    diff_angle = abs(cand_angle - base_angle)

    return np.round(diff_angle, 2)

def get_right_leg_dev(base_joints_3d, cand_joints_3d):

    brleg_vector, _ = get_right_leg_vector(base_joints_3d)
    crleg_vector, _ = get_right_leg_vector(cand_joints_3d)

    brthigh_vector, _ = get_right_thigh_vector(base_joints_3d)
    crthigh_vector, _ = get_right_thigh_vector(cand_joints_3d)

    base_angle = np.arccos(np.dot(brthigh_vector, brleg_vector)) * 180.0 / np.pi
    cand_angle = np.arccos(np.dot(crthigh_vector, crleg_vector)) * 180.0 / np.pi

    # print('base_angle ', base_angle, ' cand_angle ', cand_angle)
    
    diff_angle = abs(cand_angle - base_angle)

    return np.round(diff_angle, 2)

def calc_dev_all(base_joints_3d, cand_joints_3d):

    trunk_dev = get_trunk_dev(base_joints_3d, cand_joints_3d)

    larm_dev = get_left_arm_dev(base_joints_3d, cand_joints_3d)

    rarm_dev = get_right_arm_dev(base_joints_3d, cand_joints_3d)

    lthigh_dev = get_left_thigh_dev(base_joints_3d, cand_joints_3d)

    rthigh_dev = get_right_thigh_dev(base_joints_3d, cand_joints_3d)

    lleg_dev = get_left_leg_dev(base_joints_3d, cand_joints_3d)

    rleg_dev = get_right_leg_dev(base_joints_3d, cand_joints_3d)

    return [trunk_dev, larm_dev, rarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev]

def find_rotation_matrix(v1, v2):
    """ Find rotation matrix that aligns v1 to v2. """
    v1_normalized = _normalize(v1)
    v2_normalized = _normalize(v2)

    # Cross product and dot product
    v = np.cross(v1_normalized, v2_normalized)
    c = np.dot(v1_normalized, v2_normalized)

    # Skew-symmetric cross-product matrix of v
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Rotation matrix
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (np.linalg.norm(v)**2))

    return R

def get_hip_vector(joints_3d):

    lhip = joints_3d[4]
    rhip = joints_3d[1]

    hip_vector = _normalize(lhip - rhip)  

    return hip_vector, [lhip, rhip]

def get_shoulder_vector(joints_3d):

    lshoulder = joints_3d[11]
    rshoulder = joints_3d[14]

    shoulder_vector = _normalize(lshoulder - rshoulder)  

    return shoulder_vector, [lshoulder, rshoulder]

def align_hip(base_joints_3d, cand_joints_3d):

    bhip, _ = get_hip_vector(base_joints_3d)
    chip, _ = get_hip_vector(cand_joints_3d)

    rotation_matrix = find_rotation_matrix(chip, bhip)
    rot_cand_joints_3d = np.dot(cand_joints_3d, rotation_matrix.T)    
    
    return rot_cand_joints_3d

def align_shoulder(base_joints_3d, cand_joints_3d):

    bshoulder, _ = get_shoulder_vector(base_joints_3d)
    cshoulder, _ = get_shoulder_vector(cand_joints_3d)

    rotation_matrix = find_rotation_matrix(cshoulder, bshoulder)
    rot_cand_joints_3d = np.dot(cand_joints_3d, rotation_matrix.T)    
    
    return rot_cand_joints_3d
