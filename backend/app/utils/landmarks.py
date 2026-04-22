"""MediaPipe 468 face mesh landmark index constants for geometric feature calculation."""

# Left eye landmarks (6 points for EAR calculation)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right eye landmarks (6 points for EAR calculation)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR / curvature
# [left_corner, right_corner, top_center, bottom_center, inner_top, inner_bottom]
MOUTH_INDICES = [61, 291, 0, 17, 78, 308]

# Eyebrow landmarks
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336]

# Nose tip
NOSE_TIP_INDEX = 1

# Eye center landmarks (for normalization)
LEFT_EYE_CENTER_INDEX = 159
RIGHT_EYE_CENTER_INDEX = 386
