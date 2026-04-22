import { Layout, Menu } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  CameraOutlined,
  DatabaseOutlined,
  BarChartOutlined,
} from '@ant-design/icons';

const { Header, Content, Footer } = Layout;

const menuItems = [
  { key: '/detection', icon: <CameraOutlined />, label: '实时检测' },
  { key: '/dataset', icon: <DatabaseOutlined />, label: '数据集' },
  { key: '/results', icon: <BarChartOutlined />, label: '实验结果' },
];

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <div
          style={{
            color: '#fff',
            fontSize: 18,
            fontWeight: 'bold',
            marginRight: 40,
            cursor: 'pointer',
            whiteSpace: 'nowrap',
          }}
          onClick={() => navigate('/')}
        >
          Emotion Detection
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ flex: 1 }}
        />
      </Header>
      <Content style={{ padding: '24px 48px' }}>
        <Outlet />
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        Emotion Detection System - Based on MediaPipe & CNN
      </Footer>
    </Layout>
  );
}
