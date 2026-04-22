import { apiClient } from './client';
import type { PerformanceStats, ConfusionMatrix, LatencyStats } from '../types';

export async function getPerformance(): Promise<PerformanceStats> {
  const { data } = await apiClient.get('/api/v1/stats/performance');
  return data;
}

export async function getConfusionMatrix(): Promise<ConfusionMatrix> {
  const { data } = await apiClient.get('/api/v1/stats/confusion-matrix');
  return data;
}

export async function getLatency(): Promise<LatencyStats> {
  const { data } = await apiClient.get('/api/v1/stats/latency');
  return data;
}
