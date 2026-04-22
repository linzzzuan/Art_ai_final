import { useCallback, useEffect, useRef } from 'react';
import { Button, Row, Col, Space } from 'antd';
import { CameraOutlined, PauseCircleOutlined } from '@ant-design/icons';

import { useCamera } from '../../hooks/useCamera';
import { useMediaPipe } from '../../hooks/useMediaPipe';
import { useGeoFeatures } from '../../hooks/useGeoFeatures';
import { useEmotionInference } from '../../hooks/useEmotionInference';
import { cropFaceFromVideo } from '../../utils/image';

import CameraView from './CameraView';
import EmotionPanel from './EmotionPanel';
import StatsBar from './StatsBar';

const INFERENCE_INTERVAL_MS = 200; // ~5 FPS inference target

export default function DetectionPage() {
  const { videoRef, start, stop, active } = useCamera();
  const { landmarks, ready: mediaReady, reset: resetLandmarks } = useMediaPipe(videoRef, active);
  const geoFeatures = useGeoFeatures(landmarks);

  const inferenceMode = (import.meta.env.VITE_INFERENCE_MODE || 'http') as 'http' | 'websocket';
  const { response, predict, fps } = useEmotionInference(inferenceMode);

  const lastInferRef = useRef(0);

  // Periodic inference loop
  const doInference = useCallback(() => {
    if (!active || !landmarks || !geoFeatures || !videoRef.current) return;

    const now = performance.now();
    if (now - lastInferRef.current < INFERENCE_INTERVAL_MS) return;
    lastInferRef.current = now;

    const imageBase64 = cropFaceFromVideo(videoRef.current, landmarks);
    predict(imageBase64, geoFeatures);
  }, [active, landmarks, geoFeatures, videoRef, predict]);

  // Run inference on each landmark update
  useEffect(() => {
    doInference();
  }, [doInference]);

  const handleStop = () => {
    stop();
    resetLandmarks();
  };

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        {!active ? (
          <Button type="primary" icon={<CameraOutlined />} onClick={start}>
            开启摄像头
          </Button>
        ) : (
          <Button danger icon={<PauseCircleOutlined />} onClick={handleStop}>
            停止检测
          </Button>
        )}
      </Space>

      <Row gutter={24}>
        <Col span={14}>
          <CameraView
            videoRef={videoRef}
            landmarks={landmarks}
            response={response}
            fps={fps}
          />
          <div style={{ marginTop: 16 }}>
            <StatsBar
              geoFeatures={geoFeatures}
              inferenceMs={response?.inference_time_ms ?? null}
              fps={fps}
              mediaReady={mediaReady}
              faceDetected={!!landmarks}
            />
          </div>
        </Col>
        <Col span={10}>
          <EmotionPanel response={response} />
        </Col>
      </Row>
    </div>
  );
}
