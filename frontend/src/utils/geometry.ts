/**
 * Geometric feature calculation from 468 face landmarks.
 * Mirrors backend/app/services/geo_features.py
 */

import {
  LEFT_EYE_INDICES,
  RIGHT_EYE_INDICES,
  MOUTH_INDICES,
  LEFT_EYEBROW_INDICES,
  RIGHT_EYEBROW_INDICES,
  NOSE_TIP_INDEX,
  LEFT_EYE_CENTER_INDEX,
  RIGHT_EYE_CENTER_INDEX,
} from './landmarks';

function dist(a: number[], b: number[]): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
}

/** Eye Aspect Ratio — average of left and right. */
export function calcEAR(lm: number[][]): number {
  const earSingle = (indices: number[]) => {
    const p = indices.map((i) => lm[i]);
    const v1 = dist(p[1], p[5]);
    const v2 = dist(p[2], p[4]);
    const h = dist(p[0], p[3]);
    return h < 1e-6 ? 0 : (v1 + v2) / (2 * h);
  };
  return (earSingle(LEFT_EYE_INDICES) + earSingle(RIGHT_EYE_INDICES)) / 2;
}

/** Mouth Aspect Ratio. */
export function calcMAR(lm: number[][]): number {
  const left = lm[MOUTH_INDICES[0]];
  const right = lm[MOUTH_INDICES[1]];
  const top = lm[MOUTH_INDICES[2]];
  const bottom = lm[MOUTH_INDICES[3]];
  const h = dist(left, right);
  const v = dist(top, bottom);
  return h < 1e-6 ? 0 : v / h;
}

/** Average eyebrow-eye vertical distance. */
export function calcEyebrowEyeDist(lm: number[][]): number {
  const avgY = (indices: number[]) =>
    indices.reduce((s, i) => s + lm[i][1], 0) / indices.length;

  const leftBrowY = avgY(LEFT_EYEBROW_INDICES);
  const rightBrowY = avgY(RIGHT_EYEBROW_INDICES);
  const leftEyeY = lm[LEFT_EYE_CENTER_INDEX][1];
  const rightEyeY = lm[RIGHT_EYE_CENTER_INDEX][1];

  return (Math.abs(leftBrowY - leftEyeY) + Math.abs(rightBrowY - rightEyeY)) / 2;
}

/** Mouth corner curvature (positive = smile). */
export function calcMouthCurvature(lm: number[][]): number {
  const left = lm[MOUTH_INDICES[0]];
  const right = lm[MOUTH_INDICES[1]];
  const top = lm[MOUTH_INDICES[2]];
  const bottom = lm[MOUTH_INDICES[3]];
  const midY = (left[1] + right[1]) / 2;
  const centerY = (top[1] + bottom[1]) / 2;
  return centerY - midY;
}

/** Nasolabial fold depth — distance from nose tip to mouth center. */
export function calcNasolabialDepth(lm: number[][]): number {
  const nose = lm[NOSE_TIP_INDEX];
  const top = lm[MOUTH_INDICES[2]];
  const bottom = lm[MOUTH_INDICES[3]];
  const mouthCenterX = (top[0] + bottom[0]) / 2;
  const mouthCenterY = (top[1] + bottom[1]) / 2;
  return dist(nose, [mouthCenterX, mouthCenterY]);
}
