import { useState, useEffect, useCallback, useRef } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export function useMediaPipe(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  active: boolean,
) {
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const [ready, setReady] = useState(false);

  // Initialize FaceLandmarker once
  useEffect(() => {
    let cancelled = false;

    FilesetResolver.forVisionTasks(WASM_URL).then((vision) => {
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
      }).then((lm) => {
        if (!cancelled) {
          landmarkerRef.current = lm;
          setReady(true);
        }
      });
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // Detection loop
  useEffect(() => {
    if (!ready || !active || !videoRef.current) return;

    let running = true;

    const detect = () => {
      if (!running || !landmarkerRef.current || !videoRef.current) return;

      const video = videoRef.current;
      if (video.readyState >= 2) {
        const result = landmarkerRef.current.detectForVideo(video, performance.now());
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          setLandmarks(
            result.faceLandmarks[0].map((p) => [p.x, p.y, p.z]),
          );
        } else {
          setLandmarks(null);
        }
      }
      requestAnimationFrame(detect);
    };

    requestAnimationFrame(detect);

    return () => {
      running = false;
    };
  }, [ready, active, videoRef]);

  const reset = useCallback(() => setLandmarks(null), []);

  return { landmarks, ready, reset };
}
