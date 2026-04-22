import { apiClient } from './client';
import type { InferenceRequest, InferenceResponse } from '../types';

export async function predictEmotion(req: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiClient.post('/api/v1/inference/emotion', req);
  return data;
}

export function connectInferenceStream(): WebSocket {
  const wsUrl = import.meta.env.VITE_WS_URL || `ws://${window.location.host}`;
  return new WebSocket(`${wsUrl}/ws/v1/inference/stream`);
}
