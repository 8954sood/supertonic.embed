import test from 'node:test';
import assert from 'node:assert/strict';
import {project, parseStyle, styleJSON, resample, optimize, random, normal, nucleiRate, wavBlob} from '../web/math.mjs';

test('style file round-trips with original tensor shapes and finite values', () => {
  const ttl = project(new Float32Array(12800).fill(1), 256);
  const dp = project(new Float32Array(128).fill(1), 16);
  assert.deepEqual(parseStyle(styleJSON(ttl, dp, {})), {ttl, dp});
  const nested = styleJSON(ttl, dp, {});
  nested.style_dp.data = [[Array.from(dp)]];
  assert.deepEqual(parseStyle(nested).dp, dp);
  nested.style_dp.data = [NaN];
  assert.throws(() => parseStyle(nested));
  assert.throws(() => project([0, 0], 2));
});
test('row projection preserves independent unit spheres', () => {
  const out = project([3, 4, 1, 1], 2);
  assert.ok(Math.abs(out[0] - 0.6) < 1e-6);
  assert.ok(Math.abs(Math.hypot(...out.slice(2)) - 1) < 1e-6);
});
test('half-pixel resampling matches PyTorch interpolation geometry', () => {
  assert.deepEqual([...resample([0, 2], 2, 4)], [0, 0.5, 1.5, 2]);
  assert.deepEqual([...resample([0, 2, 4, 6], 4, 2)], [1, 5]);
});
test('SPSA improves a sphere-constrained objective, returns measured best', async () => {
  const objective = async x => (x[0] - 1) ** 2 + x[1] ** 2;
  const result = await optimize({initial: [0, 1], width: 2, objective, iterations: 200,
    rate: 0.05, perturbation: 0.01});
  assert.ok(result.loss < 0.01, `${result.loss}`);
  assert.ok(Math.abs(await objective(result.value) - result.loss) < 1e-7);
  assert.ok(Math.abs(Math.hypot(...result.value) - 1) < 1e-6);
});
test('bounded descent handles threshold and cancellation', async () => {
  const result = await optimize({initial: [1, 0], width: 2, objective: async () => 0,
    threshold: 0.24, iterations: 10, rate: 0.1, perturbation: 0.01});
  assert.equal(result.steps, 0);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(optimize({initial: [1, 0], width: 2, objective: async () => 1,
    iterations: 10, rate: 0.1, perturbation: 0.01, signal: controller.signal}), {name: 'AbortError'});
});
test('fixed seeds reproduce latent samples', () => {
  assert.deepEqual(normal(129, random(42)), normal(129, random(42)));
  assert.notDeepEqual(normal(129, random(42)), normal(129, random(43)));
});
test('rate detector rejects silence and finds spaced envelope pulses', () => {
  assert.equal(nucleiRate(new Float32Array(32000)), null);
  const signal = Float32Array.from({length: 32000}, (_, i) =>
    Math.sin(2 * Math.PI * 220 * i / 16000) * Math.exp(-(((i % 4000 - 2000) / 500) ** 2)));
  const rate = nucleiRate(signal);
  assert.ok(rate > 3 && rate < 5, `${rate}`);
});
test('downloaded audio is a valid mono PCM WAV', async () => {
  const blob = wavBlob([-2, 0, 2]);
  const b = new DataView(await blob.arrayBuffer());
  assert.equal(b.byteLength, 50); assert.equal(b.getUint32(24, true), 44100);
  assert.equal(b.getInt16(44, true), -32767); assert.equal(b.getInt16(48, true), 32767);
});
test('unlimited mode continues beyond a fixed budget and stops at target loss', async () => {
  const result = await optimize({initial: [0, 1], width: 2,
    objective: async x => (x[0] - 1) ** 2 + x[1] ** 2,
    iterations: 0, threshold: 0.01, rate: 0.05, perturbation: 0.01});
  assert.ok(result.steps > 1); assert.ok(result.loss <= 0.01);
});
