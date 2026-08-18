/* browser-only stacker; keep hot loops typed-array based */
import LibRaw from './vendor/libraw/index.js';

const clamp = (v, lo = 0, hi = 1) => v < lo ? lo : v > hi ? hi : v;
const luminance = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const RAW_EXT = /\.(dng|nef|cr2|cr3|arw|rw2|orf|raf|raw)$/i;

function progress(label, percent) { self.postMessage({ type: 'progress', label, percent }); }
function note(message) { self.postMessage({ type: 'log', message }); }

async function decode(file, maxSide = 0, target = null, gray = false) {
  if (!self.createImageBitmap || !self.OffscreenCanvas) throw new Error('This browser does not expose the image APIs needed for local processing. Try a recent Chrome, Edge, Firefox, or Safari.');
  if (!gray && RAW_EXT.test(file.name)) return decodeRaw(file, maxSide, target);
  let bitmap;
  try { bitmap = await createImageBitmap(file); } catch (_) {
    throw new Error(`Could not decode ${file.name}. Use a supported raster format or a RAW extension handled by the bundled LibRaw decoder.`);
  }
  const sourceW = bitmap.width;
  const sourceH = bitmap.height;
  let w = target?.w || sourceW;
  let h = target?.h || sourceH;
  if (!target && maxSide > 0) {
    const scale = Math.min(1, maxSide / Math.max(sourceW, sourceH));
    w = Math.max(1, Math.round(sourceW * scale));
    h = Math.max(1, Math.round(sourceH * scale));
  }
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  if (gray) {
    const data = new Float32Array(w * h);
    for (let i = 0, p = 0; i < data.length; i++, p += 4) data[i] = luminance(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255);
    return { w, h, data };
  }
  const data = new Float32Array(w * h * 3);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    data[i * 3] = rgba[p] / 255;
    data[i * 3 + 1] = rgba[p + 1] / 255;
    data[i * 3 + 2] = rgba[p + 2] / 255;
  }
  return { w, h, data };
}

function resizeRgb(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh * 3);
  for (let y = 0; y < dh; y++) {
    const sy = (y + 0.5) * sh / dh - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = (x + 0.5) * sw / dw - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const fx = sx - x0;
      const dst = (y * dw + x) * 3;
      const p00 = (y0 * sw + x0) * 3, p01 = (y0 * sw + x1) * 3, p10 = (y1 * sw + x0) * 3, p11 = (y1 * sw + x1) * 3;
      for (let c = 0; c < 3; c++) {
        const top = src[p00 + c] * (1 - fx) + src[p01 + c] * fx;
        const bottom = src[p10 + c] * (1 - fx) + src[p11 + c] * fx;
        out[dst + c] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return out;
}

async function decodeRaw(file, maxSide = 0, target = null) {
  const raw = new LibRaw();
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    await raw.open(bytes, {
      outputBps: 16,
      noAutoBright: true,
      useCameraWb: true,
      useCameraMatrix: 1,
      userQual: 3,
      highlight: 2,
      fbddNoiserd: 0,
      gamm: [1, 1],
    });
    const image = await raw.imageData();
    if (!image?.data || !image.width || !image.height) throw new Error(`LibRaw could not produce pixels for ${file.name}.`);
    const colors = Math.max(3, image.colors || 3);
    const divisor = image.bits > 8 ? 65535 : 255;
    const source = new Float32Array(image.width * image.height * 3);
    for (let i = 0; i < image.width * image.height; i++) {
      const src = i * colors, dst = i * 3;
      source[dst] = image.data[src] / divisor;
      source[dst + 1] = image.data[src + Math.min(1, colors - 1)] / divisor;
      source[dst + 2] = image.data[src + Math.min(2, colors - 1)] / divisor;
    }
    let w = image.width, h = image.height;
    if (!target && maxSide > 0) {
      const scale = Math.min(1, maxSide / Math.max(w, h));
      w = Math.max(1, Math.round(w * scale));
      h = Math.max(1, Math.round(h * scale));
    } else if (target) {
      w = target.w; h = target.h;
    }
    const data = w === image.width && h === image.height ? source : resizeRgb(source, image.width, image.height, w, h);
    return { w, h, data };
  } catch (error) {
    throw new Error(`RAW decode failed for ${file.name}: ${error?.message || error}`);
  } finally {
    raw.dispose();
  }
}

// legacy JS math below stays dormant; Python owns registration and finishing
function gray(frame) {
  const out = new Float32Array(frame.w * frame.h);
  for (let i = 0, p = 0; i < out.length; i++, p += 3) out[i] = luminance(frame.data[p], frame.data[p + 1], frame.data[p + 2]);
  return out;
}

function resizePlane(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = (y + 0.5) * sh / dh - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = (x + 0.5) * sw / dw - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const fx = sx - x0;
      out[y * dw + x] = (src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx) * (1 - fy) + (src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx) * fy;
    }
  }
  return out;
}

function planeBlur(src, w, h, radius) {
  if (radius < 1) return src.slice();
  const r = Math.max(1, Math.round(radius));
  const horizontal = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      horizontal[y * w + x] = sum / (2 * r + 1);
      sum += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += horizontal[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / (2 * r + 1);
      sum += horizontal[Math.min(h - 1, y + r + 1) * w + x] - horizontal[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

function percentile(values, q, stride = 1) {
  const sample = [];
  for (let i = 0; i < values.length; i += stride) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  if (!sample.length) return 0;
  return sample[Math.min(sample.length - 1, Math.max(0, Math.floor((sample.length - 1) * q)))];
}

function toCanvas(frame) {
  const canvas = new OffscreenCanvas(frame.w, frame.h);
  const ctx = canvas.getContext('2d');
  const rgba = new Uint8ClampedArray(frame.w * frame.h * 4);
  for (let i = 0, p = 0; i < frame.data.length; i += 3, p += 4) {
    rgba[p] = clamp(frame.data[i]) * 255;
    rgba[p + 1] = clamp(frame.data[i + 1]) * 255;
    rgba[p + 2] = clamp(frame.data[i + 2]) * 255;
    rgba[p + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, frame.w, frame.h), 0, 0);
  return canvas;
}

function warp(frame, transform) {
  const src = toCanvas(frame);
  const dst = new OffscreenCanvas(frame.w, frame.h);
  const validCanvas = new OffscreenCanvas(frame.w, frame.h);
  const ctx = dst.getContext('2d', { willReadFrequently: true });
  const validCtx = validCanvas.getContext('2d', { willReadFrequently: true });
  // invert the ref -> current transform for canvas draw
  const a = transform.cos;
  const b = -transform.sin;
  const c = transform.sin;
  const d = transform.cos;
  const cx = frame.w / 2;
  const cy = frame.h / 2;
  const tx = cx - (a * cx + c * cy) - (a * transform.dx + c * transform.dy);
  const ty = cy - (b * cx + d * cy) - (b * transform.dx + d * transform.dy);
  ctx.setTransform(a, b, c, d, tx, ty);
  ctx.drawImage(src, 0, 0);
  validCtx.setTransform(a, b, c, d, tx, ty);
  validCtx.fillStyle = '#fff';
  validCtx.fillRect(0, 0, frame.w, frame.h);
  const rgba = ctx.getImageData(0, 0, frame.w, frame.h).data;
  const coverage = validCtx.getImageData(0, 0, frame.w, frame.h).data;
  const out = new Float32Array(frame.data.length);
  const valid = new Uint8Array(frame.w * frame.h);
  for (let i = 0, p = 0; i < valid.length; i++, p += 4) {
    out[i * 3] = rgba[p] / 255;
    out[i * 3 + 1] = rgba[p + 1] / 255;
    out[i * 3 + 2] = rgba[p + 2] / 255;
    valid[i] = coverage[p + 3] > 1 ? 1 : 0;
  }
  return { w: frame.w, h: frame.h, data: out, valid };
}

function sampleBilinear(g, w, h, x, y) {
  if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) return 0;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const a = g[y0 * w + x0] * (1 - fx) + g[y0 * w + x0 + 1] * fx;
  const b = g[(y0 + 1) * w + x0] * (1 - fx) + g[(y0 + 1) * w + x0 + 1] * fx;
  return a * (1 - fy) + b * fy;
}

function downsample(g, w, h, side = 112) {
  const scale = Math.min(1, side / Math.max(w, h));
  const dw = Math.max(24, Math.round(w * scale));
  const dh = Math.max(24, Math.round(h * scale));
  return { w: dw, h: dh, data: resizePlane(g, w, h, dw, dh) };
}

function findTransform(reference, current, allowRotation) {
  const a = downsample(reference.data, reference.w, reference.h);
  const b = downsample(current.data, current.w, current.h, Math.max(a.w, a.h));
  const points = [];
  const threshold = percentile(a.data, 0.965);
  const stride = Math.max(2, Math.floor(Math.min(a.w, a.h) / 64));
  for (let y = stride; y < a.h - stride; y += stride) for (let x = stride; x < a.w - stride; x += stride) {
    const value = a.data[y * a.w + x];
    if (value >= threshold && value >= a.data[(y - 1) * a.w + x] && value >= a.data[(y + 1) * a.w + x]) points.push([x, y, value]);
  }
  points.sort((p, q) => q[2] - p[2]);
  const chosen = points.slice(0, 180);
  const angles = allowRotation ? [-2, -1.25, -0.5, 0, 0.5, 1.25, 2] : [0];
  const limitX = Math.max(3, Math.round(a.w * 0.07));
  const limitY = Math.max(3, Math.round(a.h * 0.07));
  let best = { score: -Infinity, angle: 0, dx: 0, dy: 0 };
  const cx = a.w / 2, cy = a.h / 2;
  for (const degrees of angles) {
    const rad = degrees * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    for (let dy = -limitY; dy <= limitY; dy += 1) for (let dx = -limitX; dx <= limitX; dx += 1) {
      let score = 0, weight = 0;
      for (const [x, y, value] of chosen) {
        const rx = x - cx, ry = y - cy;
        const sx = cx + cos * rx - sin * ry + dx;
        const sy = cy + sin * rx + cos * ry + dy;
        const sample = sampleBilinear(b.data, b.w, b.h, sx, sy);
        score += sample * value;
        weight += value;
      }
      const normalized = weight ? score / weight : -Infinity;
      if (normalized > best.score) best = { score: normalized, angle: rad, dx, dy };
    }
  }
  const sx = current.w / a.w;
  return { angle: best.angle, cos: Math.cos(best.angle), sin: Math.sin(best.angle), dx: best.dx * sx, dy: best.dy * sx, score: best.score };
}

function addFrame(sum, count, mean, m2, frame, valid, mode) {
  const pixels = count.length;
  for (let i = 0; i < pixels; i++) {
    if (!valid[i]) continue;
    const p = i * 3;
    const lum = luminance(frame.data[p], frame.data[p + 1], frame.data[p + 2]);
    if (mode === 'sigma' && count[i] >= 4) {
      const variance = m2[i] / Math.max(1, count[i] - 1);
      if (Math.abs(lum - mean[i]) > 3 * Math.sqrt(variance + 1e-5)) continue;
    }
    sum[p] += frame.data[p]; sum[p + 1] += frame.data[p + 1]; sum[p + 2] += frame.data[p + 2];
    const n = ++count[i];
    const delta = lum - mean[i];
    mean[i] += delta / n;
    m2[i] += delta * (lum - mean[i]);
  }
}

function finish(sum, count, mode) {
  const out = new Float32Array(sum.length);
  const target = mode === 'sum' ? Math.max(1, percentile(count, 0.55)) : 1;
  for (let i = 0; i < count.length; i++) {
    const p = i * 3;
    const scale = count[i] ? target / count[i] : 0;
    out[p] = sum[p] * scale;
    out[p + 1] = sum[p + 1] * scale;
    out[p + 2] = sum[p + 2] * scale;
  }
  return out;
}

function makeAutoMask(rawVariance, w, h) {
  const baseline = percentile(rawVariance, 0.48, 4);
  const candidate = new Float32Array(rawVariance.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    candidate[i] = y > h * 0.38 && rawVariance[i] < baseline * 0.9 ? 1 : 0;
  }
  const soft = planeBlur(candidate, w, h, Math.max(3, Math.round(Math.min(w, h) * 0.006)));
  for (let i = 0; i < soft.length; i++) soft[i] = clamp((soft[i] - 0.18) / 0.55);
  return soft;
}

function maskFromImage(mask, w, h) {
  const out = mask.w === w && mask.h === h ? mask.data.slice() : resizePlane(mask.data, mask.w, mask.h, w, h);
  const blurred = planeBlur(out, w, h, Math.max(2, Math.round(Math.min(w, h) * 0.003)));
  for (let i = 0; i < blurred.length; i++) blurred[i] = clamp(blurred[i]);
  return blurred;
}

function applyCalibration(frame, dark) {
  if (!dark) return frame;
  for (let i = 0; i < frame.data.length; i++) frame.data[i] = Math.max(0, frame.data[i] - dark[i]);
  return frame;
}

function chromaCorrect(data, w, h) {
  const green = new Float32Array(w * h);
  const red = new Float32Array(w * h);
  const blue = new Float32Array(w * h);
  for (let i = 0, p = 0; i < green.length; i++, p += 3) { red[i] = data[p]; green[i] = data[p + 1]; blue[i] = data[p + 2]; }
  const correct = (channel) => {
    let best = { score: Infinity, dx: 0, dy: 0 };
    const step = Math.max(1, Math.floor(Math.min(w, h) / 220));
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      let err = 0, n = 0;
      for (let y = 3; y < h - 3; y += step) for (let x = 3; x < w - 3; x += step) {
        const j = y * w + x, k = (y + dy) * w + x + dx;
        if (k < 0 || k >= channel.length) continue;
        const a = channel[k], b = green[j];
        if (b > 0.06 || a > 0.06) { const d = a - b; err += d * d; n++; }
      }
      if (n && err / n < best.score) best = { score: err / n, dx, dy };
    }
    if (!best.dx && !best.dy) return channel;
    const shifted = new Float32Array(channel.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) shifted[y * w + x] = channel[Math.min(h - 1, Math.max(0, y + best.dy)) * w + Math.min(w - 1, Math.max(0, x + best.dx))];
    return shifted;
  };
  const r = correct(red), b = correct(blue);
  for (let i = 0, p = 0; i < green.length; i++, p += 3) { data[p] = r[i]; data[p + 1] = green[i]; data[p + 2] = b[i]; }
}

function postProcess(data, w, h, options, mask) {
  const pixels = w * h;
  const lum = new Float32Array(pixels);
  for (let i = 0, p = 0; i < pixels; i++, p += 3) lum[i] = luminance(data[p], data[p + 1], data[p + 2]);
  if (options.lightPollution) {
    const background = planeBlur(lum, w, h, Math.max(8, Math.round(Math.min(w, h) * 0.025)));
    const base = percentile(background, 0.2, 8);
    for (let i = 0, p = 0; i < pixels; i++, p += 3) {
      const delta = Math.max(0, background[i] - base) * 0.62;
      data[p] = Math.max(0, data[p] - delta); data[p + 1] = Math.max(0, data[p + 1] - delta); data[p + 2] = Math.max(0, data[p + 2] - delta);
    }
  }
  if (options.dynamicDenoise) {
    const smooth = planeBlur(lum, w, h, Math.max(1, Math.round(Math.min(w, h) * 0.0028)));
    const detail = new Float32Array(pixels);
    const star = new Float32Array(pixels);
    const threshold = percentile(lum, 0.972, 5);
    for (let i = 0; i < pixels; i++) { detail[i] = lum[i] - smooth[i]; star[i] = clamp((lum[i] - threshold) / Math.max(0.02, 1 - threshold)); }
    const chromaR = new Float32Array(pixels), chromaB = new Float32Array(pixels);
    for (let i = 0, p = 0; i < pixels; i++, p += 3) { chromaR[i] = data[p] - data[p + 1]; chromaB[i] = data[p + 2] - data[p + 1]; }
    const blurR = planeBlur(chromaR, w, h, Math.max(2, Math.round(Math.min(w, h) * 0.004))), blurB = planeBlur(chromaB, w, h, Math.max(2, Math.round(Math.min(w, h) * 0.004)));
    for (let i = 0, p = 0; i < pixels; i++, p += 3) {
      const protect = clamp(star[i] * 1.5);
      const local = clamp(0.48 - Math.abs(detail[i]) * 7, 0.08, 0.5) * (1 - protect);
      const l = lum[i] * (1 - local) + smooth[i] * local;
      const chromaWeight = 0.58 * (1 - protect);
      const r = chromaR[i] * (1 - chromaWeight) + blurR[i] * chromaWeight;
      const b = chromaB[i] * (1 - chromaWeight) + blurB[i] * chromaWeight;
      const scale = l / Math.max(1e-5, luminance(data[p], data[p + 1], data[p + 2]));
      data[p] = Math.max(0, (data[p + 1] + r) * scale); data[p + 1] = Math.max(0, data[p + 1] * scale); data[p + 2] = Math.max(0, (data[p + 1] + b) * scale);
    }
  }
  if (options.starEnhancement) {
    const smooth = planeBlur(lum, w, h, 2);
    const threshold = percentile(lum, 0.965, 4);
    for (let i = 0, p = 0; i < pixels; i++, p += 3) {
      const star = clamp((lum[i] - threshold) / Math.max(0.025, 1 - threshold));
      const high = Math.max(0, lum[i] - smooth[i]) * star * 0.28;
      data[p] += high; data[p + 1] += high; data[p + 2] += high;
    }
  }
  if (options.hdr) {
    for (let i = 0, p = 0; i < pixels; i++, p += 3) {
      const l = luminance(data[p], data[p + 1], data[p + 2]);
      if (l <= 1e-5) continue;
      const mapped = Math.log1p(l * 2.3) / Math.log1p(2.3);
      const scale = mapped / l;
      data[p] *= scale; data[p + 1] *= scale; data[p + 2] *= scale;
    }
  }
  if (options.autoBrightness) {
    const bgMask = mask ? mask.map((v) => 1 - v) : null;
    const samples = [];
    for (let i = 0; i < pixels; i += Math.max(1, Math.floor(pixels / 50000))) if (!bgMask || bgMask[i] > 0.5) samples.push(luminance(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]));
    samples.sort((a, b) => a - b);
    const at = (q) => samples[Math.min(samples.length - 1, Math.floor((samples.length - 1) * q))] || 0;
    const black = at(0.012), white = Math.max(black + 0.02, at(0.995));
    for (let i = 0, p = 0; i < pixels; i++, p += 3) {
      const l = luminance(data[p], data[p + 1], data[p + 2]);
      const stretched = clamp((l - black) / (white - black));
      const mapped = Math.sinh(stretched * 1.55) / Math.sinh(1.55);
      const scale = l > 1e-5 ? mapped / l : 0;
      data[p] *= scale; data[p + 1] *= scale; data[p + 2] *= scale;
    }
  }
  if (options.chromaticAberration) chromaCorrect(data, w, h);
  for (let i = 0; i < data.length; i++) data[i] = clamp(data[i]);
}

function toOutput(data) {
  const out = new Uint8ClampedArray(data.length / 3 * 4);
  for (let i = 0, p = 0; i < data.length; i += 3, p += 4) { out[p] = clamp(data[i]) * 255; out[p + 1] = clamp(data[i + 1]) * 255; out[p + 2] = clamp(data[i + 2]) * 255; out[p + 3] = 255; }
  return out;
}

function packFrame(frame) {
  // LibRaw already delivered 16-bit values; this is lossless and halves the handoff memory.
  const data = new Uint16Array(frame.data.length);
  for (let index = 0; index < data.length; index++) data[index] = Math.round(clamp(frame.data[index]) * 65535);
  return { w: frame.w, h: frame.h, data };
}

function cropByCoverage(data, w, h, count, required) {
  if (required <= 1) return { data, w, h, cropped: false };
  let left = w, top = h, right = -1, bottom = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (count[y * w + x] >= required) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); }
  }
  if (right < left || bottom < top || right - left < 32 || bottom - top < 32 || (left === 0 && top === 0 && right === w - 1 && bottom === h - 1)) return { data, w, h, cropped: false };
  const out = new Float32Array((right - left + 1) * (bottom - top + 1) * 3);
  const cw = right - left + 1;
  for (let y = 0; y < bottom - top + 1; y++) for (let x = 0; x < cw; x++) {
    const src = ((y + top) * w + x + left) * 3, dst = (y * cw + x) * 3;
    out[dst] = data[src]; out[dst + 1] = data[src + 1]; out[dst + 2] = data[src + 2];
  }
  return { data: out, w: cw, h: bottom - top + 1, cropped: true };
}

let pythonWorker = null;

function ensurePythonWorker() {
  if (pythonWorker) return pythonWorker;
  pythonWorker = new Worker(new URL('./python_worker.js?v=stream314', import.meta.url), { type: 'module' });
  return pythonWorker;
}

function pythonRequest(child, payload, transfer = [], expected = ['streamReady']) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      const message = event.data || {};
      if (message.type === 'progress') progress(message.label, message.percent);
      else if (message.type === 'log') note(message.message);
      else if (message.type === 'error') {
        child.removeEventListener('message', onMessage);
        reject(new Error(message.message || 'Python processing failed'));
      } else if (expected.includes(message.type)) {
        child.removeEventListener('message', onMessage);
        resolve(message);
      }
    };
    child.addEventListener('message', onMessage);
    child.postMessage(payload, transfer);
  });
}

async function run(job) {
  const options = job.settings || {};
  const maxSide = options.resolution === 'mobile' ? 1800 : options.resolution === 'preview' ? 1100 : 0;
  // Match Python's acquisition ordering before choosing the middle reference.
  const lights = [...job.lights].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const darkFiles = [...(job.darks || [])].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const referenceIndex = Math.floor(lights.length / 2);
  progress('Decoding reference', 3);
  const referenceFrame = await decode(lights[referenceIndex], maxSide);
  const target = { w: referenceFrame.w, h: referenceFrame.h };
  const reference = packFrame(referenceFrame);
  referenceFrame.data = null;
  const darks = [];
  for (const file of darkFiles) {
    const frame = await decode(file, maxSide, target);
    const packed = packFrame(frame);
    packed.channels = 3;
    packed.dtype = 'uint16';
    darks.push(packed);
    frame.data = null;
  }
  let mask = null;
  if (job.mask) {
    const decodedMask = await decode(job.mask, maxSide, target, true);
    mask = packFrame(decodedMask);
    mask.channels = 1;
    mask.dtype = 'uint16';
    decodedMask.data = null;
  }
  note(`Native stream: ${lights.length} light frame${lights.length === 1 ? '' : 's'} at ${target.w} × ${target.h}; one frame is decoded at a time.`);
  const rawCount = lights.filter((file) => RAW_EXT.test(file.name)).length;
  if (rawCount) note(`${rawCount} RAW frame${rawCount === 1 ? '' : 's'} decoded locally with LibRaw at 16-bit, linear tone.`);

  const child = ensurePythonWorker();
  await pythonRequest(child, {
    type: 'streamStart',
    settings: options,
    count: lights.length,
    referenceIndex,
    reference: { w: reference.w, h: reference.h, channels: 3, dtype: 'uint16', buffer: reference.data.buffer },
    darks: darks.map((frame) => ({ w: frame.w, h: frame.h, channels: 3, dtype: 'uint16', buffer: frame.data.buffer })),
    mask: mask ? { w: mask.w, h: mask.h, channels: 1, dtype: 'uint16', buffer: mask.data.buffer } : null,
  }, [reference.data.buffer, ...darks.map((frame) => frame.data.buffer), ...(mask ? [mask.data.buffer] : [])]);

  // Same nearest-to-middle bootstrap as the desktop pipeline.
  const ordered = [...lights.keys()]
    .filter((index) => index !== referenceIndex)
    .sort((a, b) => Math.abs(a - referenceIndex) - Math.abs(b - referenceIndex));
  let pending = null;
  for (let done = 0; done < ordered.length; done++) {
    const index = ordered[done];
    // Decode the next RAW while Pyodide registers the current one.
    const framePromise = pending || decode(lights[index], maxSide, target);
    pending = done + 1 < ordered.length ? decode(lights[ordered[done + 1]], maxSide, target) : null;
    const frame = await framePromise;
    const packed = packFrame(frame);
    frame.data = null;
    await pythonRequest(child, {
      type: 'streamFrame', stage: 'first', index,
      frame: { w: packed.w, h: packed.h, channels: 3, dtype: 'uint16', buffer: packed.data.buffer },
    }, [packed.data.buffer]);
    progress(`Aligning and stacking ${done + 1}/${ordered.length}`, 8 + ((done + 1) / Math.max(1, ordered.length)) * 48);
  }
  const firstPass = await pythonRequest(child, { type: 'streamFirstDone' }, [], ['streamFirstReady']);
  const probeIndices = firstPass.indices || [];
  pending = null;
  for (let done = 0; done < probeIndices.length; done++) {
    const index = probeIndices[done];
    const framePromise = pending || decode(lights[index], maxSide, target);
    pending = done + 1 < probeIndices.length ? decode(lights[probeIndices[done + 1]], maxSide, target) : null;
    const frame = await framePromise;
    const packed = packFrame(frame);
    frame.data = null;
    await pythonRequest(child, {
      type: 'streamFrame', stage: 'probe', index,
      frame: { w: packed.w, h: packed.h, channels: 3, dtype: 'uint16', buffer: packed.data.buffer },
    }, [packed.data.buffer]);
  }
  await pythonRequest(child, { type: 'streamProbeDone' });
  const accepted = firstPass.accepted || [];
  pending = null;
  for (let done = 0; done < accepted.length; done++) {
    const index = accepted[done];
    if (index === referenceIndex) continue;
    const nextAccepted = accepted.slice(done + 1).find((candidate) => candidate !== referenceIndex);
    const framePromise = pending || decode(lights[index], maxSide, target);
    pending = nextAccepted === undefined ? null : decode(lights[nextAccepted], maxSide, target);
    const frame = await framePromise;
    const packed = packFrame(frame);
    frame.data = null;
    await pythonRequest(child, {
      type: 'streamFrame', stage: 'second', index,
      frame: { w: packed.w, h: packed.h, channels: 3, dtype: 'uint16', buffer: packed.data.buffer },
    }, [packed.data.buffer]);
    progress(`Robust rejection pass ${done + 1}/${Math.max(1, accepted.length - 1)}`, 60 + ((done + 1) / Math.max(1, accepted.length)) * 28);
  }
  const result = await pythonRequest(child, { type: 'streamFinish' }, [], ['pythonDone']);
  const image = new Float32Array(result.buffer);
  const output = toOutput(image);
  return { width: result.width, height: result.height, buffer: output.buffer };

}

self.onmessage = async (event) => {
  if (event.data?.type !== 'process') return;
  try {
    const result = await run(event.data);
    self.postMessage({ type: 'done', ...result }, [result.buffer]);
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message || String(error) });
  }
};
