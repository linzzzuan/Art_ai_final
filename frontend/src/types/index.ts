export type EmotionLabel = 'angry' | 'disgust' | 'fear' | 'happy' | 'sad' | 'surprise' | 'neutral';

// 推理请求
export interface InferenceRequest {
  face_image: string;
  geo_features: GeoFeatures;
}

export interface GeoFeatures {
  ear: number;
  mar: number;
  eyebrow_eye_dist: number;
  mouth_curvature: number;
  nasolabial_depth: number;
}

// 推理响应
export interface InferenceResponse {
  emotions: Record<EmotionLabel, number>;
  prediction: EmotionLabel;
  confidence: number;
  inference_time_ms: number;
}

// 数据集信息
export interface DatasetInfo {
  name: string;
  path: string;
  total_samples: number;
  image_size: string;
  num_classes: number;
  classes: EmotionLabel[];
}

export interface ClassDistribution {
  classes: Array<{ name: EmotionLabel; count: number; percentage: number }>;
}

// 分类性能
export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
}

export interface PerformanceStats {
  metrics: Record<EmotionLabel, ClassMetrics>;
  macro_avg: ClassMetrics;
  weighted_avg: ClassMetrics;
}

// 混淆矩阵
export interface ConfusionMatrix {
  labels: EmotionLabel[];
  matrix: number[][];
}

// 延迟统计
export interface LatencyStats {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  fps: number;
}

// 健康检查
export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  uptime_seconds: number;
}

// 配置信息
export interface ConfigInfo {
  active_profile: string;
  model_device: string;
  checkpoint_path: string;
  model_loaded: boolean;
}
