import { Card, Progress, Empty } from 'antd';
import type { InferenceResponse } from '../../types';
import { EMOTION_COLORS, EMOTION_LABELS_CN } from '../../constants/emotions';
import type { EmotionLabel } from '../../types';

interface Props {
  response: InferenceResponse | null;
}

export default function EmotionPanel({ response }: Props) {
  if (!response) {
    return (
      <Card title="情绪概率分布">
        <Empty description="等待检测..." />
      </Card>
    );
  }

  const sorted = Object.entries(response.emotions).sort(
    (a, b) => b[1] - a[1],
  );

  return (
    <Card title="情绪概率分布" style={{ height: '100%' }}>
      {sorted.map(([label, prob]) => (
        <div key={label} style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span>
              {EMOTION_LABELS_CN[label as EmotionLabel] || label}
            </span>
            <span style={{ fontWeight: 'bold' }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
          <Progress
            percent={Number((prob * 100).toFixed(1))}
            strokeColor={EMOTION_COLORS[label as EmotionLabel]}
            showInfo={false}
            size="small"
          />
        </div>
      ))}
    </Card>
  );
}
