import {project, random, checkAbort} from './math.mjs';

// Derivative-free, monotone pattern search on a product of unit spheres.
// Each candidate is measured against the SAME cached incumbent prompt losses.
// Screening is a cost heuristic; only the full objective can accept a move.
export async function searchTimbre({initial, width, promptCount, objective, guides = [],
  iterations = 0, threshold = 0.24, seed = 42, signal, auditInterval = 10,
  onStep = () => {}, onBest = () => {}}) {
  if (!Number.isInteger(promptCount) || promptCount < 1 || !Number.isInteger(auditInterval) || auditInterval < 1 ||
      !Number.isInteger(iterations) || iterations < 0 || width < 2) throw Error('Invalid search settings');
  const started = performance.now(), rng = random(seed);
  let x = project(initial, width), losses = [], evaluations = 0, validationCalls = 0;
  const rows = x.length / width;
  if (!rows) throw Error('Empty style');
  const presets = guides.map(g => {
    if (g.length !== x.length) throw Error('Mismatched guide dimensions');
    return project(g, width);
  });
  const evaluate = async (candidate, prompt) => {
    checkAbort(signal);
    const loss = await objective(candidate, prompt);
    evaluations++;
    checkAbort(signal);
    if (!Number.isFinite(loss)) throw Error('Non-finite objective');
    return loss;
  };
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  for (let i = 0; i < promptCount; i++) losses.push(await evaluate(x, i));
  validationCalls++;
  let bestLoss = mean(losses), initialLoss = bestLoss, improvements = 0, steps = 0, lastImprovement = 0;
  let cursor = 0, guideCursor = 0, momentum = null, stalled = 0, restarts = 0;
  const radii = {preset: 0.12, block: 0.12, momentum: 0.12};
  const failures = {preset: 0, block: 0, momentum: 0};
  const metrics = () => ({initialLoss, bestLoss, evaluations, objectiveCalls: evaluations,
    validationCalls, improvements, stepsSinceImprovement: steps - lastImprovement,
    elapsedMs: performance.now() - started, restarts, bestPromptLosses: [...losses]});
  const saveBest = () => onBest({value: x.slice(), loss: bestLoss, steps, ...metrics()});
  onStep({step: 0, loss: bestLoss, validated: true, accepted: true, ...metrics()});
  saveBest();
  for (let step = 1; (iterations === 0 || step <= iterations) && bestLoss > threshold; step++) {
    checkAbort(signal);
    // Preset directions explore known model styles, blocks allow independent
    // residuals outside their span, and successful moves receive follow-up trials.
    const kind = momentum && step % 4 === 0 ? 'momentum' : presets.length && step % 2 === 1 ? 'preset' : 'block';
    const global = kind === 'preset' && Math.floor(guideCursor / presets.length) % 2 === 0;
    const count = global || kind === 'momentum' ? rows : Math.min(rows, [1, 5, 10][Math.floor(step / 2) % 3]);
    const guide = kind === 'preset' ? presets[guideCursor++ % presets.length] : null;
    const direction = new Float32Array(x.length);
    let active = 0;
    for (let r = 0; r < count; r++) {
      const row = count === rows ? r : (cursor + r) % rows, offset = row * width;
      let dot = 0;
      for (let j = 0; j < width; j++) {
        const i = offset + j;
        direction[i] = kind === 'momentum' ? momentum[i] : guide ? guide[i] : rng() * 2 - 1;
        dot += x[i] * direction[i];
      }
      let norm = 0;
      for (let j = 0; j < width; j++) {
        const i = offset + j;
        direction[i] -= dot * x[i]; norm += direction[i] ** 2;
      }
      if (norm < 1e-12) { direction.fill(0, offset, offset + width); continue; }
      active++;
      for (let j = 0; j < width; j++) direction[offset + j] /= Math.sqrt(norm);
    }
    if (count !== rows) cursor = (cursor + count) % rows;
    // Identical presets have no tangent. Fall back to a random row so even a
    // one-step bounded run performs a real trial and every turn remains cancellable.
    if (!active) {
      const offset = cursor * width;
      let dot = 0, norm = 0;
      for (let j = 0; j < width; j++) {
        direction[offset + j] = rng() * 2 - 1;
        dot += direction[offset + j] * x[offset + j];
      }
      for (let j = 0; j < width; j++) {
        direction[offset + j] -= dot * x[offset + j]; norm += direction[offset + j] ** 2;
      }
      if (norm < 1e-12) throw Error('Degenerate search direction');
      for (let j = 0; j < width; j++) direction[offset + j] /= Math.sqrt(norm);
      cursor = (cursor + 1) % rows; active = 1;
    }
    const radius = radii[kind];
    const candidate = sign => {
      const out = x.slice();
      for (let offset = 0; offset < x.length; offset += width) {
        if (!direction.subarray(offset, offset + width).some(v => v !== 0)) continue;
        for (let j = 0; j < width; j++) {
          const i = offset + j;
          out[i] = Math.cos(radius) * x[i] + sign * Math.sin(radius) * direction[i];
        }
      }
      return project(out, width);
    };
    const prompt = (step - 1) % promptCount, plus = candidate(1), minus = candidate(-1);
    const p = await evaluate(plus, prompt), q = await evaluate(minus, prompt);
    const proposal = p <= q ? plus : minus, screenLoss = Math.min(p, q), screenBaseline = losses[prompt];
    const validated = screenLoss < losses[prompt] || step % auditInterval === 0 || step === iterations;
    let loss = null, accepted = false;
    if (validated) {
      const candidateLosses = new Array(promptCount); candidateLosses[prompt] = screenLoss;
      for (let k = 1; k < promptCount; k++) {
        const index = (prompt + k) % promptCount;
        candidateLosses[index] = await evaluate(proposal, index);
      }
      validationCalls++;
      loss = mean(candidateLosses);
      if (loss < bestLoss - 1e-9) {
        momentum = Float32Array.from(proposal, (v, i) => v - x[i]);
        x = proposal; losses = candidateLosses; bestLoss = loss;
        improvements++; lastImprovement = step; accepted = true; stalled = 0;
      }
    }
    if (accepted) {
      failures[kind] = 0; radii[kind] = Math.min(0.6, radius * 1.2);
    } else {
      stalled++;
      if (++failures[kind] >= 6) {
        radii[kind] = Math.max(0.003, radius * 0.5); failures[kind] = 0;
      }
      if (kind === 'momentum') momentum = null;
    }
    // Re-expand around the measured best instead of drifting or stopping an
    // unlimited run. A fixed ladder also revisits small and large scales.
    if (stalled >= 40) {
      const reset = [0.03, 0.3, 0.06, 0.6, 0.12][restarts++ % 5];
      for (const key of Object.keys(radii)) { radii[key] = reset; failures[key] = 0; }
      stalled = 0; momentum = null;
    }
    steps = step;
    if (accepted) saveBest();
    onStep({step, loss, validated, accepted, probeLoss: screenLoss, screenBaseline,
      direction: kind, activeRows: active, radius, ...metrics()});
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return {value: x, loss: bestLoss, steps, ...metrics()};
}
