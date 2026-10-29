import test from 'node:test';
import assert from 'node:assert/strict';
import {optimize} from '../web/math.mjs';
import {runSession, Timings} from '../web/inference.mjs';

const options = {initial: [0, 1], width: 2, rate: 0.05, perturbation: 0.01};

test('interval 5 reduces actual prompt evaluations while retaining full-objective best', async () => {
  const run = async validationInterval => {
    let evaluations = 0;
    const objective = async (x, step) => {
      evaluations += step === 0 ? 5 : 1;
      return (x[0] - 1) ** 2 + x[1] ** 2;
    };
    const result = await optimize({...options, objective, iterations: 100, validationInterval});
    assert.ok(Math.abs(await objective(result.value, 1) - result.loss) < 1e-7);
    return {result, evaluations: evaluations - 1};
  };
  const baseline = await run(1), interval = await run(5);
  assert.equal(baseline.evaluations, 705);
  assert.equal(interval.evaluations, 305);
  assert.equal(interval.result.objectiveCalls, 221);
  assert.equal(interval.result.validationCalls, 21);
  // The candidate trajectory is unchanged; only best acceptance is less frequent.
  assert.ok(interval.result.loss < 0.02);
});

test('low probe loss never stops unlimited mode before full validation confirms target', async () => {
  const events = [];
  let validations = 0;
  const result = await optimize({...options, iterations: 0, threshold: 0.24, validationInterval: 5,
    objective: async (_, step) => step ? 0 : (++validations === 1 ? 1 : 0.2),
    onStep: e => events.push(e)});
  assert.equal(result.steps, 5);
  assert.equal(result.loss, 0.2);
  assert.equal(events[1].loss, null);
  assert.equal(events[1].bestLoss, 1);
  assert.equal(events[1].validated, false);
  assert.equal(events[5].validated, true);
});

test('bounded last step is fully validated even between intervals; regressions retain best', async () => {
  let validations = 0;
  const result = await optimize({...options, iterations: 7, validationInterval: 5,
    objective: async (_, step) => step ? 0 : [1, 0.5, 0.8][validations++]});
  assert.equal(validations, 3);
  assert.equal(result.loss, 0.5);
  assert.equal(result.improvements, 1);
  assert.equal(result.stepsSinceImprovement, 2);
});

test('cancellation after an in-flight probe prevents another inference', async () => {
  const controller = new AbortController();
  let calls = 0;
  await assert.rejects(optimize({...options, iterations: 10, validationInterval: 5,
    signal: controller.signal, objective: async () => {
      if (++calls === 2) controller.abort();
      return 1;
    }}), {name: 'AbortError'});
  assert.equal(calls, 2);
  await assert.rejects(optimize({...options, validationInterval: 0}), /validation interval/);
});

function tensor(data = [1]) {
  return {dims: [1, data.length], disposed: 0,
    getData: async () => Float32Array.from(data), dispose() { this.disposed++; }};
}

test('session runner copies outputs, keeps shared inputs alive, records timings', async () => {
  const shared = tensor(), input = tensor(), outputs = [], timings = new Timings();
  const session = {run: async feeds => {
    assert.equal(feeds.shared.disposed, 0);
    const out = tensor([3]); outputs.push(out); return {out};
  }};
  const result = await runSession(session, 'vector', {shared, input}, {retained: [shared], timings});
  await runSession(session, 'vector', {shared}, {retained: [shared], timings});
  assert.deepEqual([...result.out.data], [3]);
  assert.equal(shared.disposed, 0);
  assert.equal(input.disposed, 1);
  assert.ok(outputs.every(t => t.disposed === 1));
  const report = timings.snapshot();
  assert.equal(report['vector.run'].calls, 2);
  assert.equal(report['vector.readback'].calls, 2);
  assert.ok(report['vector.run'].warmMeanMs >= 0);
});

test('failed readback waits for other outputs before cleanup; owned inputs release on run failure', async () => {
  const a = tensor(), b = tensor(), input = tensor();
  a.getData = async () => { throw Error('read failure'); };
  b.getData = async () => {
    await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(b.disposed, 0);
    return new Float32Array([2]);
  };
  await assert.rejects(runSession({run: async () => ({a, b})}, 'mock', {input}), /read failure/);
  assert.equal(a.disposed, 1); assert.equal(b.disposed, 1); assert.equal(input.disposed, 1);
  const failed = tensor(), shared = tensor();
  await assert.rejects(runSession({run: async () => { throw Error('run failure'); }}, 'mock',
    {failed, shared}, {retained: [shared]}), /run failure/);
  assert.equal(failed.disposed, 1); assert.equal(shared.disposed, 0);
});

test('cancellation during session execution still releases outputs and owned inputs', async () => {
  const controller = new AbortController(), out = tensor(), input = tensor();
  await assert.rejects(runSession({run: async () => {
    controller.abort(); return {out};
  }}, 'mock', {input}, {signal: controller.signal}), {name: 'AbortError'});
  assert.equal(out.disposed, 1); assert.equal(input.disposed, 1);
});
