import { useState, useRef, useCallback, useEffect } from 'react';
import type { InferenceResponse, GeoFeatures, EmotionLabel } from '../types';
import { predictEmotion, connectInferenceStream } from '../services/inference';

type Mode = 'http' | 'websocket';

const SMOOTHING_WINDOW = 5;
const WS_RECONNECT_DELAY_MS = 2000;
const WS_MAX_RECONNECT_ATTEMPTS = 5;

/** Apply moving-average smoothing over the last N frames. */
function smoothEmotions(
  buffer: InferenceResponse[],
): InferenceResponse {
  const latest = buffer[buffer.length - 1];
  if (buffer.length <= 1) return latest;

  const keys = Object.keys(latest.emotions) as EmotionLabel[];
  const smoothed = {} as Record<EmotionLabel, number>;

  for (const key of keys) {
    const sum = buffer.reduce((acc, r) => acc + (r.emotions[key] || 0), 0);
    smoothed[key] = sum / buffer.length;
  }

  // Find new prediction from smoothed values
  let maxProb = -1;
  let prediction = latest.prediction;
  for (const key of keys) {
    if (smoothed[key] > maxProb) {
      maxProb = smoothed[key];
      prediction = key;
    }
  }

  return {
    emotions: smoothed,
    prediction,
    confidence: maxProb,
    inference_time_ms: latest.inference_time_ms,
  };
}

export function useEmotionInference(mode: Mode = 'http') {
  const [response, setResponse] = useState<InferenceResponse | null>(null);
  const [fps, setFps] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const frameTimesRef = useRef<number[]>([]);
  const smoothingBufferRef = useRef<InferenceResponse[]>([]);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track FPS
  const recordFrame = useCallback(() => {
    const now = performance.now();
    const times = frameTimesRef.current;
    times.push(now);
    // Keep only last 30 frames
    while (times.length > 30) times.shift();
    if (times.length >= 2) {
      const elapsed = (times[times.length - 1] - times[0]) / 1000;
      setFps(Math.round((times.length - 1) / elapsed));
    }
  }, []);

  // Apply smoothing and update state
  const updateResponse = useCallback(
    (data: InferenceResponse) => {
      const buf = smoothingBufferRef.current;
      buf.push(data);
      while (buf.length > SMOOTHING_WINDOW) buf.shift();
      setResponse(smoothEmotions(buf));
      recordFrame();
    },
    [recordFrame],
  );

  // WebSocket connection with auto-reconnect
  const connectWs = useCallback(() => {
    if (mode !== 'websocket') return;

    const ws = connectInferenceStream();
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      reconnectAttemptsRef.current = 0;
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (!data.error) {
          updateResponse(data);
        }
      } catch {
        /* ignore parse errors */
      }
    };

    ws.onerror = () => {
      console.error('WebSocket inference error');
    };

    ws.onclose = () => {
      setWsConnected(false);
      wsRef.current = null;

      // Auto-reconnect
      if (reconnectAttemptsRef.current < WS_MAX_RECONNECT_ATTEMPTS) {
        reconnectAttemptsRef.current += 1;
        const delay = WS_RECONNECT_DELAY_MS * reconnectAttemptsRef.current;
        console.warn(`WebSocket closed. Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${WS_MAX_RECONNECT_ATTEMPTS})...`);
        reconnectTimerRef.current = setTimeout(connectWs, delay);
      } else {
        console.error('WebSocket max reconnect attempts reached');
      }
    };
  }, [mode, updateResponse]);

  useEffect(() => {
    if (mode !== 'websocket') return;
    connectWs();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectAttemptsRef.current = WS_MAX_RECONNECT_ATTEMPTS; // prevent reconnect on unmount
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [mode, connectWs]);

  // HTTP predict
  const predict = useCallback(
    async (imageBase64: string, geoFeatures: GeoFeatures) => {
      if (mode === 'websocket') {
        // Send via WebSocket
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ face_image: imageBase64, geo_features: geoFeatures }),
          );
        }
      } else {
        // HTTP POST
        try {
          const result = await predictEmotion({
            face_image: imageBase64,
            geo_features: geoFeatures,
          });
          updateResponse(result);
        } catch (err) {
          console.error('Inference request failed:', err);
        }
      }
    },
    [mode, updateResponse],
  );

  return { response, predict, fps, wsConnected };
}
