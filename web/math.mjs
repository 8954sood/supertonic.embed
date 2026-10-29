// Numerical parts deliberately do not depend on the browser or ONNX runtime.
export function random(seed = 42) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function normal(n, rng) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12)));
    const t = 2 * Math.PI * rng();
    out[i] = r * Math.cos(t);
    if (i + 1 < n) out[i + 1] = r * Math.sin(t);
  }
  return out;
}
export function project(x, width) {
  const out = Float32Array.from(x);
  if (out.length % width) throw Error('Invalid style row width');
  for (let i = 0; i < out.length; i += width) {
    let sum = 0;
    for (let j = 0; j < width; j++) sum += out[i + j] ** 2;
    if (!Number.isFinite(sum) || sum < 1e-16) throw Error('Invalid style row');
    const norm = Math.sqrt(sum);
    for (let j = 0; j < width; j++) out[i + j] /= norm;
  }
  return out;
}
export function parseStyle(json) {
  const read = (key, dims) => {
    const obj = json[key];
    if (!obj || JSON.stringify(obj.dims) !== JSON.stringify(dims)) throw Error(`Invalid ${key} dimensions`);
    const data = Float32Array.from(obj.data.flat(Infinity));
    if (data.length !== dims.reduce((a, b) => a * b) || !data.every(Number.isFinite)) throw Error(`Invalid ${key} data`);
    return data;
  };
  return {ttl: read('style_ttl', [1, 50, 256]), dp: read('style_dp', [1, 8, 16])};
}
export function styleJSON(ttl, dp, metadata) {
  return {style_ttl: {data: Array.from(ttl), dims: [1, 50, 256], type: 'float32'},
    style_dp: {data: Array.from(dp), dims: [1, 8, 16], type: 'float32'}, metadata};
}
export function mse(a, b) {
  if (a.length !== b.length || !a.length) throw Error('Mismatched objective dimensions');
  return a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
}
export const statsLoss = (a, b) => mse(a.mean, b.mean) + mse(a.std, b.std);
export function resample(x, from, to) {
  const n = Math.round(x.length * to / from), out = new Float32Array(n);
  // PyTorch interpolate(mode='linear', align_corners=False) coordinates.
  for (let i = 0; i < n; i++) {
    const pos = Math.max(0, (i + 0.5) * x.length / n - 0.5);
    const lo = Math.min(Math.floor(pos), x.length - 1), hi = Math.min(lo + 1, x.length - 1);
    out[i] = x[lo] + (x[hi] - x[lo]) * (pos - lo);
  }
  return out;
}
export function nucleiRate(x, sr = 16000) {
  // Browser approximation of librosa trim/RMS + scipy peak prominence.
  const frames = [];
  for (let c = 0; c < x.length; c += 160) {
    let sum = 0;
    for (let j = c - 200; j < c + 200; j++) sum += (x[j] || 0) ** 2;
    frames.push(Math.sqrt(sum / 400));
  }
  const peak = Math.max(0, ...frames);
  if (peak < 1e-8) return null;
  const gate = peak * 10 ** (-35 / 20);
  const first = frames.findIndex(v => v >= gate), last = frames.findLastIndex(v => v >= gate);
  const seconds = (last - first + 1) * 160 / sr;
  if (seconds < 0.3) return null;
  const env = frames.slice(first, last + 1).map((_, i, a) =>
    [-2, -1, 0, 1, 2].reduce((s, j) => s + (a[i + j] || 0), 0) / 5);
  const candidates = [];
  for (let i = 1; i < env.length - 1; i++) {
    if (env[i] <= env[i - 1] || env[i] < env[i + 1]) continue;
    let left = env[i], right = env[i];
    for (let j = i - 1; j >= 0 && env[j] <= env[i]; j--) left = Math.min(left, env[j]);
    for (let j = i + 1; j < env.length && env[j] <= env[i]; j++) right = Math.min(right, env[j]);
    if (env[i] - Math.max(left, right) >= peak * 0.08) candidates.push(i);
  }
  candidates.sort((a, b) => env[b] - env[a]);
  const accepted = [];
  for (const i of candidates) if (accepted.every(j => Math.abs(i - j) >= 10)) accepted.push(i);
  return accepted.length ? accepted.length / seconds : null;
}
export function checkAbort(signal) { if (signal?.aborted) throw new DOMException('취소됨', 'AbortError'); }

// Two-sided simultaneous perturbation; both probes must use the same prompt/noise.
// Acceptance is measured separately; a noisy update is never assumed to be better.
export async function optimize({initial, width, objective, iterations = 0, rate, perturbation,
  seed = 42, signal, onStep = () => {}, threshold = -Infinity}) {
  checkAbort(signal);
  const rng = random(seed), anchor = project(initial, width);
  let x = anchor, best = x.slice(), bestLoss = await objective(x, 0);
  if (!Number.isFinite(bestLoss)) throw Error('Non-finite objective');
  const m = new Float32Array(x.length), v = new Float32Array(x.length);
  let steps = 0;
  onStep({step: 0, loss: bestLoss});
  for (let step = 1; (iterations === 0 || step <= iterations) && bestLoss > threshold; step++) {
    checkAbort(signal);
    const c = perturbation / step ** 0.101;
    const delta = Float32Array.from(x, () => rng() < 0.5 ? -1 : 1);
    const plus = project(x.map((a, i) => a + c * delta[i]), width);
    const minus = project(x.map((a, i) => a - c * delta[i]), width);
    const p = await objective(plus, step), q = await objective(minus, step);
    const slope = (p - q) / (2 * c);
    if (!Number.isFinite(slope)) throw Error('Non-finite objective');
    const lr = rate / (1 + step / (iterations || 1000)) ** 0.602;
    x = project(x.map((a, i) => {
      const g = Math.max(-1, Math.min(1, slope * delta[i]));
      m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g;
      return a - lr * (m[i] / (1 - 0.9 ** step)) / (Math.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8);
    }), width);
    const loss = await objective(x, 0); // fixed validation prompts, comparable across iterations
    if (!Number.isFinite(loss)) throw Error('Non-finite objective');
    if (loss < bestLoss) { bestLoss = loss; best = x.slice(); }
    steps = step;
    onStep({step, loss, bestLoss});
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return {value: best, loss: bestLoss, steps};
}
export function wavBlob(samples, sr = 44100) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buffer);
  const str = (pos, s) => [...s].forEach((c, i) => v.setUint8(pos + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, buffer.byteLength - 8, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => v.setInt16(44 + 2 * i, Math.round(Math.max(-1, Math.min(1, x)) * 32767), true));
  return new Blob([buffer], {type: 'audio/wav'});
}
