# Phase 5: 前端实时检测

## 1. 目标

实现前端摄像头采集 → MediaPipe 关键点检测 → 几何特征计算 → 后端推理 → 结果可视化的完整流程。

## 2. MediaPipe 集成

### 2.1 安装依赖

```bash
npm install @mediapipe/tasks-vision
```

### 2.2 `useMediaPipe` Hook

**文件**: `frontend/src/hooks/useMediaPipe.ts`

```typescript
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export function useMediaPipe(videoRef: React.RefObject<HTMLVideoElement>) {
  const [landmarker, setLandmarker] = useState<FaceLandmarker | null>(null);
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);

  // 初始化
  useEffect(() => {
    FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    ).then((vision) => {
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
      }).then(setLandmarker);
    });
  }, []);

  // 持续检测
  useEffect(() => {
    if (!landmarker || !videoRef.current) return;

    let running = true;
    const detect = () => {
      if (!running) return;
      const result = landmarker.detectForVideo(videoRef.current, performance.now());
      if (result.faceLandmarks.length > 0) {
        setLandmarks(result.faceLandmarks[0].map((p) => [p.x, p.y, p.z]));
      }
      requestAnimationFrame(detect);
    };
    requestAnimationFrame(detect);

    return () => { running = false; };
  }, [landmarker, videoRef]);

  return landmarks;
}
```

## 3. 几何特征计算

### 3.1 `useGeoFeatures` Hook

**文件**: `frontend/src/hooks/useGeoFeatures.ts`

```typescript
import { normalizeLandmarks } from '@/utils/landmarks';
import { calcEAR, calcMAR, calcEyebrowEyeDist, calcMouthCurvature, calcNasolabialDepth } from '@/utils/geometry';

export function useGeoFeatures(landmarks: number[][] | null) {
  return useMemo(() => {
    if (!landmarks) return null;
    const normalized = normalizeLandmarks(landmarks);
    return {
      ear: calcEAR(normalized),
      mar: calcMAR(normalized),
      eyebrow_eye_dist: calcEyebrowEyeDist(normalized),
      mouth_curvature: calcMouthCurvature(normalized),
      nasolabial_depth: calcNasolabialDepth(normalized),
    };
  }, [landmarks]);
}
```

### 3.2 几何计算函数

**文件**: `frontend/src/utils/geometry.ts`

```typescript
// 眼宽高比
export function calcEAR(lm: number[][]): number {
  // 使用左右眼垂直/水平距离比
  const leftEyeH = dist(lm[159], lm[145]);  // 上眼睑到下眼睑
  const leftEyeW = dist(lm[33], lm[133]);   // 内眼角到外眼角
  const rightEyeH = dist(lm[386], lm[374]);
  const rightEyeW = dist(lm[362], lm[263]);
  const leftEAR = leftEyeH / leftEyeW;
  const rightEAR = rightEyeH / rightEyeW;
  return (leftEAR + rightEAR) / 2;
}

// 嘴宽高比
export function calcMAR(lm: number[][]): number {
  const mouthW = dist(lm[61], lm[291]);   // 嘴角到嘴角
  const mouthH = dist(lm[0], lm[17]);     // 上唇到下唇
  return mouthW / mouthH;
}

// 眉眼间距
export function calcEyebrowEyeDist(lm: number[][]): number {
  const leftBrowCenter = avg(lm[70], lm[105]);
  const leftEyeCenter = avg(lm[33], lm[133]);
  const rightBrowCenter = avg(lm[300], lm[334]);
  const rightEyeCenter = avg(lm[362], lm[263]);
  return (dist(leftBrowCenter, leftEyeCenter) + dist(rightBrowCenter, rightEyeCenter)) / 2;
}

// 嘴角曲率（基于嘴角与相邻唇部点的角度）
export function calcMouthCurvature(lm: number[][]): number {
  // 嘴角上扬程度：比较嘴角与唇中心的 y 坐标差
  const lipCenterY = (lm[0].y + lm[17].y) / 2;
  const leftMouthY = lm[61].y;
  const rightMouthY = lm[291].y;
  return ((lipCenterY - leftMouthY) + (lipCenterY - rightMouthY)) / 2;
}

// 鼻唇沟深度
export function calcNasolabialDepth(lm: number[][]): number {
  const noseLeft = lm[116];
  const noseRight = lm[345];
  const mouthLeft = lm[61];
  const mouthRight = lm[291];
  const leftDist = dist(noseLeft, mouthLeft);
  const rightDist = dist(noseRight, mouthRight);
  return (leftDist + rightDist) / 2;
}

function dist(a: number[], b: number[]): number {
  return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2);
}

function avg(a: number[], b: number[]): number[] {
  return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];
}
```

### 3.3 关键点归一化

**文件**: `frontend/src/utils/landmarks.ts`

```typescript
export function normalizeLandmarks(landmarks: number[][]): number[][] {
  // 双眼中心
  const leftEye = landmarks[33];   // 左眼内眼角
  const rightEye = landmarks[362]; // 右眼内眼角
  const cx = (leftEye[0] + rightEye[0]) / 2;
  const cy = (leftEye[1] + rightEye[1]) / 2;
  // 眼距
  const eyeDist = Math.sqrt((rightEye[0] - leftEye[0])**2 + (rightEye[1] - leftEye[1])**2);

  return landmarks.map(([x, y, z]) => [(x - cx) / eyeDist, (y - cy) / eyeDist, z / eyeDist]);
}
```

## 4. 摄像头管理

**文件**: `frontend/src/hooks/useCamera.ts`

```typescript
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [active, setActive] = useState(false);

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setActive(true);
    }
  };

  const stop = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(t => t.stop());
    setActive(false);
  };

  return { videoRef, start, stop, active };
}
```

## 5. 推理请求

**文件**: `frontend/src/hooks/useEmotionInference.ts`

```typescript
export function useEmotionInference(geoFeatures: GeoFeatures | null) {
  const [response, setResponse] = useState<InferenceResponse | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // HTTP 模式
  const predictHTTP = async (imageBase64: string) => {
    if (!geoFeatures) return;
    const result = await predictEmotion({ face_image: imageBase64, geo_features: geoFeatures });
    setResponse(result);
  };

  // WebSocket 模式
  useEffect(() => {
    const ws = connectInferenceStream();
    wsRef.current = ws;

    ws.onmessage = (e) => {
      setResponse(JSON.parse(e.data));
    };

    return () => ws.close();
  }, []);

  const predictWS = (imageBase64: string) => {
    if (!geoFeatures || !wsRef.current) return;
    wsRef.current.send(JSON.stringify({ face_image: imageBase64, geo_features: geoFeatures }));
  };

  return { response, predictHTTP, predictWS };
}
```

## 6. 图像裁剪与编码

**文件**: `frontend/src/utils/image.ts`

```typescript
// 根据关键点边界框裁剪面部
export function cropFace(
  canvas: HTMLCanvasElement,
  landmarks: number[][]
): string {
  const ctx = canvas.getContext('2d')!;
  // 计算边界框
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  landmarks.forEach(([x, y]) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  });

  // 扩大边界框 20%
  const w = maxX - minX, h = maxY - minY;
  const cx = minX + w/2, cy = minY + h/2;
  const scale = 1.2;
  const newW = w * scale, newH = h * scale;

  // 裁剪并缩放到 224x224
  const faceCanvas = document.createElement('canvas');
  faceCanvas.width = 224;
  faceCanvas.height = 224;
  const faceCtx = faceCanvas.getContext('2d')!;
  faceCtx.drawImage(canvas,
    cx - newW/2, cy - newH/2, newW, newH,  // 源区域
    0, 0, 224, 224                          // 目标区域
  );

  return faceCanvas.toDataURL('image/jpeg', 0.85);
}
```

## 7. 检测页面

### 7.1 CameraView 组件

```typescript
// frontend/src/pages/Detection/CameraView.tsx
export function CameraView({ videoRef, landmarks, response }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // 绘制 MediaPipe 关键点网格
  useEffect(() => {
    if (!landmarks || !overlayRef.current) return;
    drawLandmarks(overlayRef.current, landmarks);
  }, [landmarks]);

  return (
    <div style={{ position: 'relative' }}>
      <video ref={videoRef} style={{ width: 640, height: 480, transform: 'scaleX(-1)' }} />
      <canvas ref={overlayRef} style={{ position: 'absolute', top: 0, left: 0, width: 640, height: 480 }} />
      {response && (
        <div style={{ position: 'absolute', top: 10, left: 10, fontSize: 24, color: '#fff', textShadow: '1px 1px 3px #000' }}>
          {response.prediction} ({(response.confidence * 100).toFixed(1)}%)
        </div>
      )}
    </div>
  );
}
```

### 7.2 EmotionPanel 组件

```typescript
// frontend/src/pages/Detection/EmotionPanel.tsx
export function EmotionPanel({ response }: Props) {
  if (!response) return <Empty description="等待检测..." />;

  return (
    <Card title="情绪概率分布">
      {Object.entries(response.emotions)
        .sort((a, b) => b[1] - a[1])
        .map(([label, prob]) => (
          <Progress
            key={label}
            percent={prob * 100}
            strokeColor={EMOTION_COLORS[label]}
            format={() => `${label} ${(prob * 100).toFixed(1)}%`}
          />
        ))}
    </Card>
  );
}
```

## 8. 交付物

- [ ] `useCamera` Hook — 摄像头开启/关闭
- [ ] `useMediaPipe` Hook — 468 关键点实时检测
- [ ] `useGeoFeatures` Hook — 5 维几何特征计算
- [ ] `useEmotionInference` Hook — HTTP/WS 推理
- [ ] `cropFace` 函数 — 面部裁剪 + base64 编码
- [ ] CameraView 组件 — 视频 + 关键点叠加 + 情绪标签
- [ ] EmotionPanel 组件 — 七类情绪概率柱状图
- [ ] Detection 页面整合完成
- [ ] 端到端检测流畅，FPS ≥ 10
