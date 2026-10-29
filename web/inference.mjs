import {checkAbort} from './math.mjs';

// Wall-clock measurements, not GPU kernel time or evidence of operator placement.
export class Timings {
  constructor() { this.entries = {}; }
  add(name, ms) {
    const entry = this.entries[name] ??= {calls: 0, totalMs: 0, firstMs: ms, warmMs: 0};
    entry.calls++; entry.totalMs += ms;
    if (entry.calls > 1) entry.warmMs += ms;
  }
  snapshot() {
    return Object.fromEntries(Object.entries(this.entries).map(([name, e]) => [name, {...e,
      warmMeanMs: e.calls > 1 ? e.warmMs / (e.calls - 1) : null}]));
  }
}

export async function runSession(session, name, feeds, {signal, timings, retained = []} = {}) {
  let output;
  const borrowed = new Set(retained);
  try {
    checkAbort(signal);
    const start = performance.now();
    output = await session.run(feeds);
    timings?.add(`${name}.run`, performance.now() - start);
    checkAbort(signal);
    const readStart = performance.now();
    // Wait for *all* reads even on failure, before releasing any GPU outputs.
    const reads = await Promise.allSettled(Object.entries(output).map(async ([key, tensor]) =>
      [key, {data: Float32Array.from(await tensor.getData()), dims: [...tensor.dims]}]));
    timings?.add(`${name}.readback`, performance.now() - readStart);
    checkAbort(signal);
    const failed = reads.find(r => r.status === 'rejected');
    if (failed) throw failed.reason;
    return Object.fromEntries(reads.map(r => r.value));
  } finally {
    for (const tensor of new Set(Object.values(output ?? {}))) tensor.dispose();
    for (const tensor of new Set(Object.values(feeds))) if (!borrowed.has(tensor)) tensor.dispose();
  }
}
