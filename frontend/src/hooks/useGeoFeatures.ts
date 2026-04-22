import { useMemo } from 'react';
import { normalizeLandmarks } from '../utils/landmarks';
import {
  calcEAR,
  calcMAR,
  calcEyebrowEyeDist,
  calcMouthCurvature,
  calcNasolabialDepth,
} from '../utils/geometry';
import type { GeoFeatures } from '../types';

export function useGeoFeatures(landmarks: number[][] | null): GeoFeatures | null {
  return useMemo(() => {
    if (!landmarks || landmarks.length < 468) return null;
    const norm = normalizeLandmarks(landmarks);
    return {
      ear: calcEAR(norm),
      mar: calcMAR(norm),
      eyebrow_eye_dist: calcEyebrowEyeDist(norm),
      mouth_curvature: calcMouthCurvature(norm),
      nasolabial_depth: calcNasolabialDepth(norm),
    };
  }, [landmarks]);
}
