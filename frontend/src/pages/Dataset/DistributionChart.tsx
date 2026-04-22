import ReactECharts from 'echarts-for-react';
import type { ClassDistribution } from '../../types';
import { EMOTION_COLORS, EMOTION_LABELS_CN } from '../../constants/emotions';
import type { EmotionLabel } from '../../types';

interface Props {
  classes: ClassDistribution['classes'];
}

export default function DistributionChart({ classes }: Props) {
  const option = {
    xAxis: {
      type: 'category',
      data: classes.map(c => EMOTION_LABELS_CN[c.name as EmotionLabel] || c.name),
      axisLabel: { fontSize: 13 },
    },
    yAxis: { type: 'value', name: '样本数' },
    series: [
      {
        type: 'bar',
        data: classes.map(c => ({
          value: c.count,
          itemStyle: { color: EMOTION_COLORS[c.name as EmotionLabel] || '#1890ff' },
        })),
        label: { show: true, position: 'top', formatter: '{c}' },
        barMaxWidth: 50,
      },
    ],
    tooltip: { trigger: 'axis' },
    grid: { top: 40, bottom: 40, left: 60, right: 20 },
  };

  return <ReactECharts option={option} style={{ height: 400 }} />;
}
