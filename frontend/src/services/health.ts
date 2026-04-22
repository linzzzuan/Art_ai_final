import { apiClient } from './client';
import type { HealthStatus, ConfigInfo } from '../types';

export async function getHealth(): Promise<HealthStatus> {
  const { data } = await apiClient.get('/health');
  return data;
}

export async function getConfigInfo(): Promise<ConfigInfo> {
  const { data } = await apiClient.get('/api/v1/config');
  return data;
}
