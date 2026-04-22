# Phase 8: 优化与打磨

## 1. 目标

优化性能、完善错误处理、编写文档。

## 2. 推理性能优化

### 2.1 模型量化（可选）

```python
# 加载权重后量化
model.cnn = torch.quantization.quantize_dynamic(
    model.cnn, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

### 2.2 前端优化

- 请求节流：限制推理频率匹配目标 FPS
- 结果平滑：移动平均窗口（默认 5 帧）
- 降级策略：网络断开时显示提示，自动重连

### 2.3 后端优化

- Gunicorn workers 数量调优（GPU 推理建议 1-2 workers）
- 图像解码优化（使用 PIL 预加载）
- 内存泄漏检查

## 3. 错误处理

### 3.1 前端错误边界

```typescript
// 全局错误捕获
window.addEventListener('error', (e) => {
  console.error('Global error:', e.error);
});

// 组件级错误边界
class ErrorBoundary extends React.Component {
  // ...
}
```

### 3.2 后端异常中间件

```python
@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)

@app.exception_handler(RuntimeError)
async def runtime_error(request, exc):
    return JSONResponse({"detail": "Internal server error"}, status_code=500)
```

### 3.3 WebSocket 异常处理

- 连接断开自动重连
- 推理失败返回错误 JSON 而非断开连接

## 4. 文档

### 4.1 README.md

```markdown
# Emotion Detection System

基于 MediaPipe + CNN 的多模态实时情绪检测系统。

## 快速开始

### 本地开发
1. `pip install -r requirements.txt`
2. `cd backend && uvicorn app.main:app --reload`
3. `cd frontend && npm install && npm run dev`

### 云端部署
1. `./scripts/deploy_setup.sh`
2. `./scripts/run_training.sh --task-name v1`
3. `./scripts/start_server.sh`

## API 文档
启动后端后访问 http://localhost:8000/docs

## 配置
编辑 `config.yaml` 修改环境配置。
```

### 4.2 API 文档

FastAPI 自动生成 Swagger UI（`/docs`）和 ReDoc（`/redoc`）。

## 5. 交付物

- [ ] 推理性能达标（CPU ≥ 10 FPS, GPU ≥ 30 FPS）
- [ ] 前端错误边界完善
- [ ] 后端异常中间件完善
- [ ] WebSocket 自动重连
- [ ] README.md 编写完成
- [ ] 所有 Phase 测试通过
