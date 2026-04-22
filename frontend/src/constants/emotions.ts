import type { EmotionLabel } from '../types';

export const EMOTION_LABELS: EmotionLabel[] = [
  'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral',
];

export const EMOTION_COLORS: Record<EmotionLabel, string> = {
  angry: '#ff4d4f',
  disgust: '#a0d911',
  fear: '#722ed1',
  happy: '#faad14',
  sad: '#1890ff',
  surprise: '#13c2c2',
  neutral: '#8c8c8c',
};

export const EMOTION_LABELS_CN: Record<EmotionLabel, string> = {
  angry: '愤怒',
  disgust: '厌恶',
  fear: '恐惧',
  happy: '高兴',
  sad: '悲伤',
  surprise: '惊讶',
  neutral: '中性',
};
