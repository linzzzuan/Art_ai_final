import ReactECharts from 'echarts-for-react';
import { EMOTION_LABELS_CN } from '../../constants/emotions';
import type { EmotionLabel } from '../../types';

interface Props {
  labels: string[];
  matrix: number[][];
}

export default function ConfusionMatrix({ labels, matrix }: Props) {
  const cnLabels = labels.map(l => EMOTION_LABELS_CN[l as EmotionLabel] || l);

  const data: [number, number, number][] = [];
  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      data.push([j, i, val]);
    });
  });

  const maxVal = Math.max(...matrix.flat(), 1);

  const option = {
    tooltip: {
      position: 'top',
      formatter: (params: { data: [number, number, number] }) => {
        const [x, y, v] = params.data;
        return `预测: ${cnLabels[x]}<br/>实际: ${cnLabels[y]}<br/>数量: ${v}`;
      },
    },
    xAxis: {
      type: 'category',
      data: cnLabels,
      name: '预测标签',
      nameLocation: 'center',
      nameGap: 35,
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category',
      data: cnLabels,
      name: '实际标签',
      nameLocation: 'center',
      nameGap: 55,
      splitArea: { show: true },
    },
    visualMap: {
      min: 0,
      max: maxVal,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#f0f9ff', '#1d39c4', '#0d47a1'] },
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: true, fontSize: 12 },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' },
        },
      },
    ],
    grid: { top: 20, bottom: 80, left: 80, right: 20 },
  };

  return <ReactECharts option={option} style={{ height: 500 }} />;
}
