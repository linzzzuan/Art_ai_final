import { Card, Descriptions, Tag } from 'antd';
import type { GeoFeatures } from '../../types';

interface Props {
  geoFeatures: GeoFeatures | null;
  inferenceMs: number | null;
  fps: number;
  mediaReady: boolean;
  faceDetected: boolean;
}

export default function StatsBar({ geoFeatures, inferenceMs, fps, mediaReady, faceDetected }: Props) {
  return (
    <Card title="实时统计" size="small">
      <Descriptions column={2} size="small">
        <Descriptions.Item label="MediaPipe">
          <Tag color={mediaReady ? 'green' : 'orange'}>
            {mediaReady ? '就绪' : '加载中'}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="人脸检测">
          <Tag color={faceDetected ? 'green' : 'red'}>
            {faceDetected ? '已检测' : '未检测到'}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="FPS">{fps}</Descriptions.Item>
        <Descriptions.Item label="推理延迟">
          {inferenceMs !== null ? `${inferenceMs}ms` : '-'}
        </Descriptions.Item>
        {geoFeatures && (
          <>
            <Descriptions.Item label="EAR">{geoFeatures.ear.toFixed(3)}</Descriptions.Item>
            <Descriptions.Item label="MAR">{geoFeatures.mar.toFixed(3)}</Descriptions.Item>
            <Descriptions.Item label="眉眼间距">{geoFeatures.eyebrow_eye_dist.toFixed(3)}</Descriptions.Item>
            <Descriptions.Item label="嘴角曲率">{geoFeatures.mouth_curvature.toFixed(3)}</Descriptions.Item>
          </>
        )}
      </Descriptions>
    </Card>
  );
}
