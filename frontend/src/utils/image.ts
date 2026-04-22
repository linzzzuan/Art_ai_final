/**
 * Image utilities: face cropping and base64 encoding.
 */

/**
 * Crop face region from a video/canvas using landmark bounding box,
 * resize to 224x224, return as base64 JPEG data URI.
 */
export function cropFaceFromVideo(
  video: HTMLVideoElement,
  landmarks: number[][],
): string {
  // Calculate bounding box from landmarks (coordinates are 0-1 normalized)
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const [x, y] of landmarks) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  // Expand bounding box by 20%
  const w = (maxX - minX) * vw;
  const h = (maxY - minY) * vh;
  const cx = (minX + (maxX - minX) / 2) * vw;
  const cy = (minY + (maxY - minY) / 2) * vh;
  const scale = 1.2;
  const cropW = w * scale;
  const cropH = h * scale;

  const sx = Math.max(0, cx - cropW / 2);
  const sy = Math.max(0, cy - cropH / 2);
  const sw = Math.min(cropW, vw - sx);
  const sh = Math.min(cropH, vh - sy);

  // Draw cropped region to 224x224 canvas
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);

  return canvas.toDataURL('image/jpeg', 0.85);
}
