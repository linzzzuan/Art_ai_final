import { useEffect, useRef } from 'react';
import type { InferenceResponse } from '../../types';
import { EMOTION_COLORS } from '../../constants/emotions';

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  landmarks: number[][] | null;
  response: InferenceResponse | null;
  fps: number;
}

/** Draw 468 face mesh landmarks on the overlay canvas. */
function drawLandmarks(canvas: HTMLCanvasElement, landmarks: number[][], videoWidth: number, videoHeight: number) {
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw landmark points
  ctx.fillStyle = 'rgba(0, 255, 128, 0.5)';
  for (const [x, y] of landmarks) {
    const px = x * videoWidth;
    const py = y * videoHeight;
    ctx.beginPath();
    ctx.arc(px, py, 1, 0, Math.PI * 2);
    ctx.fill();
  }
}

export default function CameraView({ videoRef, landmarks, response, fps }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!overlayRef.current || !videoRef.current) return;
    if (landmarks) {
      const vw = videoRef.current.videoWidth || 640;
      const vh = videoRef.current.videoHeight || 480;
      drawLandmarks(overlayRef.current, landmarks, vw, vh);
    }
  }, [landmarks, videoRef]);

  const predColor = response
    ? EMOTION_COLORS[response.prediction as keyof typeof EMOTION_COLORS] || '#fff'
    : '#fff';

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <video
        ref={videoRef as React.LegacyRef<HTMLVideoElement>}
        style={{
          width: 640,
          height: 480,
          transform: 'scaleX(-1)',
          background: '#000',
          borderRadius: 8,
        }}
        muted
        playsInline
      />
      <canvas
        ref={overlayRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: 640,
          height: 480,
          transform: 'scaleX(-1)',
          pointerEvents: 'none',
        }}
      />
      {/* Prediction overlay */}
      {response && (
        <div
          style={{
            position: 'absolute',
            top: 12,
            left: 12,
            fontSize: 22,
            fontWeight: 'bold',
            color: predColor,
            textShadow: '1px 1px 4px rgba(0,0,0,0.8)',
          }}
        >
          {response.prediction} ({(response.confidence * 100).toFixed(1)}%)
        </div>
      )}
      {/* FPS / inference time */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          fontSize: 14,
          color: '#0f0',
          textShadow: '1px 1px 3px rgba(0,0,0,0.8)',
        }}
      >
        FPS: {fps}
        {response && ` | ${response.inference_time_ms}ms`}
      </div>
    </div>
  );
}
