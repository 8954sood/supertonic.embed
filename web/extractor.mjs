import * as ort from 'onnxruntime-web/webgpu';
import wasmURL from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import {random, normal, parseStyle, styleJSON, statsLoss, resample, optimize,
  nucleiRate, checkAbort, mse} from './math.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {wasm: wasmURL};
const float = (data, dims) => new ort.Tensor('float32', data, dims);
const ones = n => new Float32Array(n).fill(1);
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`${url}: HTTP ${response.status} (모델 준비 스크립트를 실행하세요)`);
  return response.json();
}

export class Extractor {
  constructor(log, signal) { this.log = log; this.signal = signal; this.sessions = {}; }
  async load(base) {
    if (!isSecureContext || !navigator.gpu) throw Error('WebGPU를 지원하는 브라우저에서 localhost 또는 HTTPS로 열어 주세요.');
    if (!await navigator.gpu.requestAdapter()) throw Error('사용 가능한 WebGPU 어댑터가 없습니다.');
    const url = name => `${base.replace(/\/$/, '')}/${name}`;
    this.cfg = await json(url('manifest.json'));
    this.geometry = await json(url('tts.json'));
    const {ttl, ae} = this.geometry;
    this.chunk = ae.base_chunk_size * ttl.chunk_compress_factor;
    this.dim = ttl.latent_dim * ttl.chunk_compress_factor;
    this.presets = [];
    for (const name of this.cfg.presets) {
      checkAbort(this.signal);
      this.presets.push({name, ...parseStyle(await json(url(`${name}.json`)))});
    }
    for (const name of ['duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder', 'wavlm_stats']) {
      checkAbort(this.signal);
      this.log(`${name} 불러오는 중…`);
      this.sessions[name] = await ort.InferenceSession.create(url(`${name}.onnx`), {
        executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all',
      });
    }
    this.prompts = [];
    const sizing = this.presets.find(p => p.name === 'M1');
    if (!sizing) throw Error('M1 preset이 없습니다.');
    const rng = random(this.cfg.seed);
    for (const prompt of this.cfg.prompts) {
      const seconds = await this.seconds(prompt, sizing.dp);
      const frames = this.frames(seconds);
      this.prompts.push({...prompt, frames, latent: normal(this.dim * frames, rng)});
    }
    this.log('모델 준비 완료. WebGPU 사용, 미지원 연산은 WASM에서 실행합니다.');
  }
  frames(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 30) throw Error(`안전한 합성 길이를 벗어났습니다: ${seconds}s`);
    return Math.max(4, Math.ceil(Math.floor(seconds * this.cfg.sampleRate) / this.chunk));
  }
  textFeeds(prompt) {
    const n = prompt.ids.length;
    return {text_ids: new ort.Tensor('int64', BigInt64Array.from(prompt.ids, BigInt), [1, n]),
      text_mask: float(ones(n), [1, 1, n])};
  }
  async run(name, feeds) {
    try {
      checkAbort(this.signal);
      const output = await this.sessions[name].run(feeds);
      try {
        checkAbort(this.signal);
        // Copy outputs before disposing ORT tensors; no GPU resources survive a call.
        return Object.fromEntries(await Promise.all(Object.entries(output).map(async ([key, tensor]) =>
          [key, {data: Float32Array.from(await tensor.getData()), dims: [...tensor.dims]}])));
      } finally { Object.values(output).forEach(t => t.dispose()); }
    } finally { Object.values(feeds).forEach(t => t.dispose()); }
  }
  async seconds(prompt, dp) {
    const result = await this.run('duration_predictor', {...this.textFeeds(prompt), style_dp: float(dp, [1, 8, 16])});
    return Object.values(result)[0].data[0] / this.cfg.speed;
  }
  async synthesize(prompt, ttl, dp, ownLength = false) {
    let frames = prompt.frames, latent = prompt.latent.slice();
    if (ownLength) {
      frames = this.frames(await this.seconds(prompt, dp));
      latent = normal(this.dim * frames, random(this.cfg.seed));
    }
    const text = Object.values(await this.run('text_encoder', {...this.textFeeds(prompt), style_ttl: float(ttl, [1, 50, 256])}))[0];
    for (let step = 0; step < this.cfg.steps; step++) {
      const output = await this.run('vector_estimator', {
        noisy_latent: float(latent, [1, this.dim, frames]), text_emb: float(text.data, text.dims),
        style_ttl: float(ttl, [1, 50, 256]), latent_mask: float(ones(frames), [1, 1, frames]),
        text_mask: float(ones(prompt.ids.length), [1, 1, prompt.ids.length]),
        current_step: float(new Float32Array([step]), [1]),
        total_step: float(new Float32Array([this.cfg.steps]), [1]),
      });
      latent = Object.values(output)[0].data;
    }
    const output = await this.run('vocoder', {latent: float(latent, [1, this.dim, frames])});
    return Object.values(output)[0].data;
  }
  async statistics(audio) {
    if (audio.length < 4800) throw Error('참조 음성은 최소 0.3초 이상이어야 합니다.');
    const out = await this.run('wavlm_stats', {audio: float(audio, [1, audio.length])});
    return {mean: out.mean.data, std: out.std.data};
  }
  async durations(dp) {
    // ORT's asyncify WASM runtime must not receive overlapping runs on a session.
    const result = [];
    for (const prompt of this.prompts) result.push(await this.seconds(prompt, dp));
    return result;
  }
  async rate(ttl, dp) {
    const rates = [];
    for (const [index, p] of this.prompts.entries()) {
      this.log(`말하기 속도 측정 · 문장 ${index + 1}/${this.prompts.length} 합성 중…`);
      const wave = await this.synthesize(p, ttl, dp, true);
      const rate = nucleiRate(resample(wave, this.cfg.sampleRate, 16000));
      if (rate !== null) rates.push(rate);
      this.log(`문장 ${index + 1}/${this.prompts.length} 완료 · ${rate === null ? "측정 불가" : rate.toFixed(3) + " 음절/초"}`);
    }
    return rates.length ? rates.reduce((a, b) => a + b) / rates.length : null;
  }
  async extract(reference, {iterations = 0, durationIterations = 100, onStep = () => {}} = {}) {
    const target = await this.statistics(reference);
    let nearest;
    for (const preset of this.presets) {
      const wav = await this.synthesize(this.prompts[0], preset.ttl, preset.dp);
      const loss = statsLoss(await this.statistics(resample(wav, this.cfg.sampleRate, 16000)), target);
      this.log(`${preset.name}: ${loss.toFixed(5)}`);
      if (!nearest || loss < nearest.loss) nearest = {...preset, loss};
    }
    this.log(`시작 프리셋: ${nearest.name}. SPSA로 음색 최적화 중…`);
    // Validate against all five prompts, rather than comparing losses from changing sentences.
    const objective = async (ttl, step) => {
      const prompts = step === 0 ? this.prompts : [this.prompts[(step - 1) % this.prompts.length]];
      let sum = 0;
      for (const prompt of prompts) {
        const wav = await this.synthesize(prompt, ttl, nearest.dp);
        sum += statsLoss(await this.statistics(resample(wav, this.cfg.sampleRate, 16000)), target);
      }
      return sum / prompts.length;
    };
    const timbre = await optimize({initial: nearest.ttl, width: 256, objective, iterations,
      rate: 2e-4, perturbation: 0.002, signal: this.signal, seed: this.cfg.seed,
      onStep: e => onStep({stage: 'timbre', ...e}), threshold: 0.24});
    let dp = nearest.dp.slice();
    const duration = {status: 'reference unmeasurable', rate_ref: nucleiRate(reference), path: []};
    if (duration.rate_ref) {
      this.log('참조 음성의 말하기 속도에 맞추는 중…');
      const start = await this.rate(timbre.value, dp);
      duration.rate_start = start;
      if (start) {
        this.log('기준 문장 길이 측정 중…');
        const base = await this.durations(nearest.dp);
        let bestError = Math.abs(start / duration.rate_ref - 1), rho = 1;
        let bestRate = start;
        duration.path.push([1, start]); duration.status = 'iteration limit';
        for (let iter = 0; iter < 8; iter++) {
          if (bestError < 0.03) { duration.status = 'converged'; break; }
          rho *= Math.max(0.5, Math.min(2, bestRate / duration.rate_ref));
          const targets = base.map(v => v * rho);
          const fit = await optimize({initial: nearest.dp, width: 16, iterations: durationIterations,
            rate: 0.005, perturbation: 0.01, seed: this.cfg.seed + iter, signal: this.signal,
            objective: async candidate => {
              const predicted = await this.durations(candidate);
              // Prior is sum of squared distances, as in the Python objective.
              return mse(predicted, targets) + 0.01 * mse(candidate, nearest.dp) * candidate.length;
            }, onStep: e => {
              onStep({stage: `duration ${iter + 1}/8`, ...e});
              if (e.step % 10 === 0) this.log(`속도 보정 ${iter + 1}/8 · 최적화 ${e.step}/${durationIterations}`);
            }});
          const rate = await this.rate(timbre.value, fit.value);
          if (!rate) { duration.status = 'synthesis unmeasurable'; break; }
          duration.path.push([rho, rate]);
          const error = Math.abs(rate / duration.rate_ref - 1);
          if (error >= bestError) { duration.status = 'no improvement'; break; }
          bestError = error; dp = fit.value; bestRate = rate;
        }
        duration.rate_final = bestRate;
        if (bestError < 0.03) duration.status = 'converged';
      } else duration.status = 'starting synthesis unmeasurable';
    }
    const report = {method: 'webgpu-spsa-experimental', preset: nearest.name,
      timbre_loss: timbre.loss, timbre_steps: timbre.steps,
      timbre_status: timbre.loss <= 0.24 ? "threshold reached" : "iteration limit", duration,
      settings: {iterations, durationIterations, seed: this.cfg.seed, samplingSteps: this.cfg.steps},
      executionProviders: ['webgpu', 'wasm'],
      note: 'Gradient-free approximation. Original CUDA/Adam quality and speed are not established.'};
    const audio = await this.synthesize(this.prompts[0], timbre.value, dp, true);
    return {style: styleJSON(timbre.value, dp, {extracted_at: new Date().toISOString(), ...report}), report, audio};
  }
  async dispose() {
    for (const session of Object.values(this.sessions)) await session.release();
    this.sessions = {};
  }
}
