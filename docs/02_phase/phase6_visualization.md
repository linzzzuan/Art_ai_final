# Phase 6: 数据分析与可视化

## 1. 目标

实现数据集信息页面和实验结果可视化页面。

## 2. 数据集页面

### 2.1 Dataset 页面

**文件**: `frontend/src/pages/Dataset/index.tsx`

- 调用 `GET /api/v1/dataset/info` 显示数据集概览
- 调用 `GET /api/v1/dataset/class-distribution` 显示类别分布柱状图

### 2.2 DistributionChart 组件

```typescript
// frontend/src/pages/Dataset/DistributionChart.tsx
import ReactECharts from 'echarts-for-react';

export function DistributionChart({ classes }: Props) {
  const option = {
    xAxis: {
      type: 'category',
      data: classes.map(c => c.name),
    },
    yAxis: { type: 'value', name: '样本数' },
    series: [{
      type: 'bar',
      data: classes.map(c => ({ value: c.count, itemStyle: { color: EMOTION_COLORS[c.name] } })),
      label: { show: true, position: 'top', formatter: '{c}' },
    }],
    tooltip: { trigger: 'axis' },
  };

  return <ReactECharts option={option} style={{ height: 400 }} />;
}
```

## 3. 实验结果页面

### 3.1 Results 页面

**文件**: `frontend/src/pages/Results/index.tsx`

- 调用 `GET /api/v1/stats/performance` 显示分类性能表格
- 调用 `GET /api/v1/stats/confusion-matrix` 显示混淆矩阵热力图

### 3.2 PerformanceChart 组件

```typescript
// frontend/src/pages/Results/PerformanceChart.tsx
export function PerformanceChart({ metrics }: Props) {
  const labels = Object.keys(metrics);
  const f1Scores = labels.map(l => metrics[l].f1);

  const option = {
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'value', min: 0, max: 1, name: 'F1-score' },
    series: [{
      type: 'bar',
      data: f1Scores.map((v, i) => ({ value: v, itemStyle: { color: EMOTION_COLORS[labels[i]] } })),
      label: { show: true, position: 'top', formatter: '{c:.2f}' },
    }],
  };

  return <ReactECharts option={option} style={{ height: 350 }} />;
}
```

### 3.3 ConfusionMatrix 组件

```typescript
// frontend/src/pages/Results/ConfusionMatrix.tsx
export function ConfusionMatrix({ labels, matrix }: Props) {
  const data: [number, number, number][] = [];
  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      data.push([j, i, val]);
    });
  });

  const option = {
    tooltip: { position: 'top' },
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'category', data: labels },
    visualMap: { min: 0, max: Math.max(...matrix.flat()), calculable: true, inRange: { color: ['#f0f9ff', '#0d47a1'] } },
    series: [{
      type: 'heatmap',
      data,
      label: { show: true },
    }],
    grid: { top: 30, bottom: 60, left: 80 },
  };

  return <ReactECharts option={option} style={{ height: 500 }} />;
}
```

## 4. 交付物

- [ ] Dataset 页面 — 数据集概览 + 类别分布柱状图
- [ ] Results 页面 — 分类性能 F1 柱状图
- [ ] ConfusionMatrix 组件 — 7×7 混淆矩阵热力图
- [ ] 所有图表使用 ECharts 渲染
