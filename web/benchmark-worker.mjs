import {Extractor} from './extractor.mjs';
import {searchTimbre} from './search.mjs';
import {optimize, statsLoss, resample} from './math.mjs';

self.onmessage = async ({data}) => {
  const extractor = new Extractor(progress => self.postMessage({progress}));
  try {
    if (!Number.isInteger(data.budget) || data.budget < 10) throw Error('Invalid budget');
    await extractor.load('/models');
    const target = await extractor.statistics(data.reference), preset = extractor.presets.find(p => p.name === 'M1');
    const results = [];
    const budgetEnd = new Error('Evaluation budget exhausted');
    for (const method of data.first === 'pattern' ? ['pattern', 'legacy'] : ['legacy', 'pattern']) {
      let evaluations = 0, latest, bestLoss = Infinity;
      const started = performance.now(), history = [];
      const measure = async (ttl, index) => {
        if (evaluations >= data.budget) throw budgetEnd;
        const wave = await extractor.synthesize(extractor.prompts[index], ttl, preset.dp);
        const loss = statsLoss(await extractor.statistics(resample(wave, extractor.cfg.sampleRate, 16000)), target);
        evaluations++;
        return loss;
      };
      const onStep = e => {
        latest = e;
        bestLoss = e.bestLoss;
        history.push({step: e.step, evaluations, elapsedMs: performance.now() - started, bestLoss});
        self.postMessage({progress: `${method}: ${evaluations}/${data.budget}, best ${bestLoss.toFixed(6)}`});
      };
      try {
        if (method === 'pattern') await searchTimbre({initial: preset.ttl, width: 256,
          promptCount: extractor.prompts.length, guides: extractor.presets.map(p => p.ttl),
          objective: measure, onStep, threshold: -Infinity, seed: extractor.cfg.seed});
        else await optimize({initial: preset.ttl, width: 256, rate: 2e-4, perturbation: 0.002,
          seed: extractor.cfg.seed, validationInterval: 5, onStep,
          objective: async (ttl, step) => {
            if (step) return measure(ttl, (step - 1) % extractor.prompts.length);
            let total = 0;
            for (let i = 0; i < extractor.prompts.length; i++) total += await measure(ttl, i);
            return total / extractor.prompts.length;
          }});
      } catch (e) { if (e !== budgetEnd) throw e; }
      results.push({method, budget: data.budget, evaluations, elapsedMs: performance.now() - started,
        bestLoss, completedSteps: latest?.step, initialLoss: latest?.initialLoss, history});
    }
    await extractor.dispose();
    self.postMessage({result: {reference: 'examples/01.wav (CC0)', seed: extractor.cfg.seed,
      initialization: 'M1, fixed dp and prompt latents; preset-ranking cost excluded',
      note: 'Single reference/seed; finite budget can end inside a candidate evaluation. Best includes completed full validations only. Order/cache can affect wall time.',
      results}});
  } catch (e) {
    try { await extractor.dispose(); } catch {}
    self.postMessage({error: e.message});
  }
};
