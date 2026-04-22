# Phase 1: 基础架构搭建

## 1. 目标

搭建后端 FastAPI 项目骨架和前端 React 项目骨架，打通前后端基础联调通路。

## 2. 后端项目初始化

### 2.1 目录结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── health.py
│   ├── services/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── core/
│       ├── __init__.py
│       └── logging.py
├── checkpoints/
├── experiments/
├── data/
├── logs/
├── datasets/
│   └── mock/
├── config.yaml
├── requirements.txt
├── gunicorn_conf.py
└── README.md
```

### 2.2 核心文件

**`app/config.py`** — 配置加载器
- 读取 `config.yaml`
- 根据 `active_profile` 返回对应环境的配置字典
- 解析相对路径为绝对路径

**`app/main.py`** — FastAPI 应用入口
- 创建 FastAPI 实例
- 注册路由（health）
- CORS 中间件（从 config 读取 `cors_origins`）
- `/health` 端点（Phase 1 仅返回固定 JSON）

**`app/dependencies.py`** — 依赖注入
- `get_config()` — 提供配置实例
- Phase 2+ 追加 `get_model_service()`

**`gunicorn_conf.py`** — Gunicorn 配置
- workers = 4（从环境变量读取）
- worker_class = `uvicorn.workers.UvicornWorker`
- bind = `0.0.0.0:8000`

### 2.3 接口实现（Phase 1）

**`GET /health`** — 固定响应：
```json
{
  "status": "ok",
  "model_loaded": false,
  "device": "cpu",
  "uptime_seconds": 0
}
```

**`GET /api/v1/config`** — 返回当前激活的 profile 名称和模型加载状态。

### 2.4 本地启动

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# 访问 http://localhost:8000/health
# 访问 http://localhost:8000/docs 查看 Swagger UI
```

## 3. 前端项目初始化

### 3.1 创建项目

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install antd echarts echarts-for-react axios
```

### 3.2 目录结构

```
frontend/
├── public/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── pages/
│   │   ├── Detection/
│   │   │   └── index.tsx
│   │   ├── Dataset/
│   │   │   └── index.tsx
│   │   └── Results/
│   │       └── index.tsx
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── AppLayout.tsx
│   │   │   └── AppHeader.tsx
│   │   └── Charts/
│   │       └── index.tsx
│   ├── services/
│   │   ├── client.ts
│   │   └── health.ts
│   ├── types/
│   │   └── index.ts
│   └── constants/
│       └── emotions.ts
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

### 3.3 核心文件

**`vite.config.ts`** — 开发代理
```typescript
export default defineConfig({
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':  { target: 'ws://localhost:8000', ws: true },
    }
  }
})
```

**`src/services/client.ts`** — Axios 实例
```typescript
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
});
```

**`src/constants/emotions.ts`** — 情绪常量
```typescript
export const EMOTION_LABELS = [
  'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
] as const;

export const EMOTION_COLORS: Record<string, string> = {
  angry: '#ff4d4f',
  disgust: '#a0d911',
  fear: '#722ed1',
  happy: '#faad14',
  sad: '#1890ff',
  surprise: '#13c2c2',
  neutral: '#8c8c8c',
};
```

**`src/components/Layout/AppLayout.tsx`** — 布局框架
- Ant Design Layout
- Header：Logo + 导航（实时检测 / 数据集 / 实验结果）
- Content：`<Outlet />`

### 3.4 本地启动

```bash
cd frontend
npm run dev
# 访问 http://localhost:5173
```

## 4. 前后端联调

1. 后端启动：`uvicorn app.main:app --reload`
2. 前端启动：`npm run dev`
3. 前端页面调用 `GET /api/v1/config`，验证代理通路
4. 前端页面调用 `GET /health`，验证基础接口

## 5. 交付物

- [ ] 后端 FastAPI 项目可启动，`/health` 返回正确 JSON
- [ ] 前端 React 项目可启动，页面显示基础布局
- [ ] 前后端通过 Vite 代理联调成功
- [ ] `config.yaml` 包含 local/server 两个 profile
- [ ] 代码提交到 git
