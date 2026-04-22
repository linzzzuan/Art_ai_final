/**
 * MediaPipe 468 face mesh landmark index constants.
 * Mirrors backend/app/utils/landmarks.py
 */

// Left eye (6 points for EAR)
export const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
// Right eye (6 points for EAR)
export const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
// Mouth: [left_corner, right_corner, top_center, bottom_center, inner_top, inner_bottom]
export const MOUTH_INDICES = [61, 291, 0, 17, 78, 308];
// Eyebrows
export const LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107];
export const RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336];
// Nose tip
export const NOSE_TIP_INDEX = 1;
// Eye centers for normalization
export const LEFT_EYE_CENTER_INDEX = 159;
export const RIGHT_EYE_CENTER_INDEX = 386;

/** Normalize landmarks: origin = midpoint of eye centers, scale = inter-eye distance. */
export function normalizeLandmarks(landmarks: number[][]): number[][] {
  const left = landmarks[LEFT_EYE_CENTER_INDEX];
  const right = landmarks[RIGHT_EYE_CENTER_INDEX];
  const cx = (left[0] + right[0]) / 2;
  const cy = (left[1] + right[1]) / 2;
  const eyeDist = Math.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2);
  const scale = eyeDist < 1e-6 ? 1e-6 : eyeDist;

  return landmarks.map(([x, y, z]) => [
    (x - cx) / scale,
    (y - cy) / scale,
    z / scale,
  ]);
}
