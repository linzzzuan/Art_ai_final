import { useEffect, useState } from 'react';
import { Card, Descriptions, Spin, Alert, Row, Col } from 'antd';
import { getDatasetInfo, getClassDistribution } from '../../services/dataset';
import type { DatasetInfo, ClassDistribution } from '../../types';
import DistributionChart from './DistributionChart';

export default function DatasetPage() {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [dist, setDist] = useState<ClassDistribution | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [infoRes, distRes] = await Promise.all([
          getDatasetInfo(),
          getClassDistribution(),
        ]);
        setInfo(infoRes);
        setDist(distRes);
      } catch (err) {
        setError('无法获取数据集信息，请确认后端服务已启动。');
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
        <Spin size="large" tip="加载数据集信息..." />
      </div>
    );
  }

  if (error) {
    return <Alert type="warning" message={error} showIcon style={{ marginBottom: 16 }} />;
  }

  return (
    <div>
      <Row gutter={24}>
        <Col span={10}>
          <Card title="数据集概览">
            {info && (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="数据集名称">{info.name}</Descriptions.Item>
                <Descriptions.Item label="数据路径">{info.path}</Descriptions.Item>
                <Descriptions.Item label="总样本数">{info.total_samples}</Descriptions.Item>
                <Descriptions.Item label="图像尺寸">{info.image_size}</Descriptions.Item>
                <Descriptions.Item label="类别数">{info.num_classes}</Descriptions.Item>
                <Descriptions.Item label="类别列表">
                  {info.classes.join(', ')}
                </Descriptions.Item>
              </Descriptions>
            )}
          </Card>
        </Col>
        <Col span={14}>
          <Card title="类别分布">
            {dist && <DistributionChart classes={dist.classes} />}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
