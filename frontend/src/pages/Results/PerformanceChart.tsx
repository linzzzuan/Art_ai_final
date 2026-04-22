import ReactECharts from 'echarts-for-react';
import type { ClassMetrics } from '../../types';
import { EMOTION_COLORS, EMOTION_LABELS_CN } from '../../constants/emotions';
import type { EmotionLabel } from '../../types';

interface Props {
  metrics: Record<string, ClassMetrics>;
}

export default function PerformanceChart({ metrics }: Props) {
  const labels = Object.keys(metrics);
  const f1Scores = labels.map(l => metrics[l].f1);

  const option = {
    xAxis: {
      type: 'category',
      data: labels.map(l => EMOTION_LABELS_CN[l as EmotionLabel] || l),
      axisLabel: { fontSize: 13 },
    },
    yAxis: { type: 'value', min: 0, max: 1, name: 'F1-score' },
    series: [
      {
        type: 'bar',
        data: f1Scores.map((v, i) => ({
          value: v,
          itemStyle: { color: EMOTION_COLORS[labels[i] as EmotionLabel] || '#1890ff' },
        })),
        label: {
          show: true,
          position: 'top',
          formatter: (params: { value: number }) => params.value.toFixed(2),
        },
        barMaxWidth: 50,
      },
    ],
    tooltip: { trigger: 'axis' },
    grid: { top: 40, bottom: 40, left: 60, right: 20 },
  };

  return <ReactECharts option={option} style={{ height: 350 }} />;
}
