import test from 'node:test';
import assert from 'node:assert/strict';
import {searchTimbre} from '../web/search.mjs';
import {project} from '../web/math.mjs';

test('guided search improves all-prompt objective and only keeps fully measured improvements', async () => {
  const initial = [0, 1, 0, 1], target = [1, 0, 1, 0], bests = [];
  const objective = async (x, i) => (1 + i) * x.reduce((s, v, j) => s + (v - target[j]) ** 2, 0);
  const result = await searchTimbre({initial, width: 2, promptCount: 3, guides: [target],
    objective, iterations: 50, threshold: -Infinity, onBest: e => bests.push(e)});
  assert.ok(result.loss < 0.01, `${result.loss}`);
  assert.ok(bests.length > 2);
  for (let i = 1; i < bests.length; i++) assert.ok(bests[i].loss < bests[i - 1].loss);
  const measured = (await Promise.all([0, 1, 2].map(i => objective(result.value, i)))).reduce((a, b) => a + b) / 3;
  assert.ok(Math.abs(measured - result.loss) < 1e-10);
  for (let row = 0; row < 2; row++) assert.ok(Math.abs(Math.hypot(...result.value.slice(row * 2, row * 2 + 2)) - 1) < 1e-6);
});

test('candidate that improves screening prompt but worsens full mean is rejected without drift', async () => {
  const initial = [0, 1], calls = [], events = [];
  const result = await searchTimbre({initial, width: 2, guides: [[1, 0]], promptCount: 2,
    objective: async (x, i) => { calls.push({x: [...x], i}); return i ? 1 + 4 * x[0] : 1 - x[0]; },
    iterations: 1, threshold: -Infinity, onStep: e => events.push(e)});
  assert.equal(result.loss, 1);
  assert.deepEqual([...result.value], initial);
  assert.equal(events[1].accepted, false);
  assert.equal(events[1].validated, true);
  assert.equal(calls.length, 5); // Two initial, two probes, one remaining validation.
  assert.ok(calls.slice(2).every(c => Math.abs(Math.hypot(...c.x) - 1) < 1e-6));
});

test('all rows remain reachable outside the preset span, and rejected candidates do not accumulate', async () => {
  const initial = new Float32Array(50 * 4);
  for (let i = 0; i < 50; i++) initial[i * 4] = 1;
  const visited = new Set();
  const result = await searchTimbre({initial, width: 4, promptCount: 1, iterations: 100,
    threshold: -Infinity, objective: async x => {
      let changes = 0;
      for (let row = 0; row < 50; row++) {
        if (x[row * 4] !== 1) { visited.add(row); changes++; }
      }
      assert.ok(changes <= 10);
      return 1;
    }});
  assert.equal(visited.size, 50);
  assert.deepEqual(result.value, initial);
  assert.ok(result.restarts >= 2);
});

test('unlimited mode stops only when a completed full validation reaches target', async () => {
  let calls = 0;
  const result = await searchTimbre({initial: [0, 1], width: 2, promptCount: 2,
    guides: [[1, 0]], threshold: 0.24,
    objective: async x => { calls++; return x[0] > 0 ? 0.2 : 1; }});
  assert.equal(result.steps, 1);
  assert.equal(result.loss, 0.2);
  assert.equal(calls, 5);
});

test('initial target avoids trials; cancellation and non-finite objective cannot publish a partial best', async () => {
  const controller = new AbortController();
  let saves = 0, calls = 0;
  await assert.rejects(searchTimbre({initial: [0, 1], width: 2, promptCount: 2,
    signal: controller.signal, onBest: () => saves++, objective: async () => {
      if (++calls === 4) controller.abort(); return 1;
    }}), {name: 'AbortError'});
  assert.equal(saves, 1);
  assert.equal(calls, 4);
  await assert.rejects(searchTimbre({initial: [0, 1], width: 2, promptCount: 1,
    objective: async () => NaN}), /Non-finite/);
  const result = await searchTimbre({initial: [0, 1], width: 2, promptCount: 1,
    objective: async () => 0.2});
  assert.equal(result.steps, 0);
});

test('identical guide falls back safely, seed reproduces trials and input arrays stay unchanged', async () => {
  const initial = project([1, 2, 3, 4], 4), copy = initial.slice();
  const run = () => searchTimbre({initial, width: 4, promptCount: 1, guides: [initial], iterations: 8,
    threshold: -Infinity, objective: async x => 1 - x[0]});
  const a = await run(), b = await run();
  assert.deepEqual(a.value, b.value);
  assert.deepEqual(initial, copy);
  assert.equal(a.steps, 8);
  assert.equal(a.evaluations, b.evaluations);
});
