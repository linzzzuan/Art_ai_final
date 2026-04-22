import { useEffect, useState } from 'react';
import { Card, Table, Spin, Alert, Row, Col, Statistic } from 'antd';
import { getPerformance, getConfusionMatrix } from '../../services/stats';
import type { PerformanceStats, ConfusionMatrix as ConfusionMatrixType, ClassMetrics } from '../../types';
import { EMOTION_LABELS_CN } from '../../constants/emotions';
import type { EmotionLabel } from '../../types';
import PerformanceChart from './PerformanceChart';
import ConfusionMatrixComp from './ConfusionMatrix';

export default function ResultsPage() {
  const [perf, setPerf] = useState<PerformanceStats | null>(null);
  const [cm, setCm] = useState<ConfusionMatrixType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [perfRes, cmRes] = await Promise.all([
          getPerformance(),
          getConfusionMatrix(),
        ]);
        setPerf(perfRes);
        setCm(cmRes);
      } catch (err) {
        setError('无法获取实验结果，请确认后端服务已启动且已完成训练。');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 80 }}>
        <Spin size="large" tip="加载实验结果..." />
      </div>
    );
  }

  if (error) {
    return <Alert type="warning" message={error} showIcon style={{ marginBottom: 16 }} />;
  }

  // Build table data
  const tableData = perf
    ? Object.entries(perf.metrics).map(([label, m]: [string, ClassMetrics]) => ({
        key: label,
        label: EMOTION_LABELS_CN[label as EmotionLabel] || label,
        precision: m.precision.toFixed(3),
        recall: m.recall.toFixed(3),
        f1: m.f1.toFixed(3),
      }))
    : [];

  const columns = [
    { title: '类别', dataIndex: 'label', key: 'label' },
    { title: 'Precision', dataIndex: 'precision', key: 'precision' },
    { title: 'Recall', dataIndex: 'recall', key: 'recall' },
    { title: 'F1-score', dataIndex: 'f1', key: 'f1' },
  ];

  return (
    <div>
      {/* Summary stats */}
      {perf && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={8}>
            <Card>
              <Statistic
                title="Macro F1"
                value={perf.macro_avg.f1}
                precision={3}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="Weighted F1"
                value={perf.weighted_avg.f1}
                precision={3}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="Weighted Precision"
                value={perf.weighted_avg.precision}
                precision={3}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={24}>
        {/* Performance table + chart */}
        <Col span={12}>
          <Card title="分类性能详情" style={{ marginBottom: 24 }}>
            <Table
              dataSource={tableData}
              columns={columns}
              pagination={false}
              size="small"
            />
          </Card>
          {perf && (
            <Card title="F1-score 分布">
              <PerformanceChart metrics={perf.metrics} />
            </Card>
          )}
        </Col>

        {/* Confusion matrix */}
        <Col span={12}>
          {cm && (
            <Card title="混淆矩阵">
              <ConfusionMatrixComp labels={cm.labels} matrix={cm.matrix} />
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
}
