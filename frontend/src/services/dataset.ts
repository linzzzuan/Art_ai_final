import { apiClient } from './client';
import type { DatasetInfo, ClassDistribution } from '../types';

export async function getDatasetInfo(): Promise<DatasetInfo> {
  const { data } = await apiClient.get('/api/v1/dataset/info');
  return data;
}

export async function getClassDistribution(): Promise<ClassDistribution> {
  const { data } = await apiClient.get('/api/v1/dataset/class-distribution');
  return data;
}
