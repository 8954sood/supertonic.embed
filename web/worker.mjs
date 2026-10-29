import {Extractor} from './extractor.mjs';

self.onmessage = async ({data}) => {
  if (data.type !== 'extract') return;
  const extractor = new Extractor(message => self.postMessage({type: 'log', message}));
  try {
    await extractor.load(data.base);
    const result = await extractor.extract(data.reference, {...data.options,
      onBest: style => self.postMessage({type: 'best', style}),
      onStep: step => self.postMessage({type: 'step', step})});
    await extractor.dispose();
    self.postMessage({type: 'done', result}, [result.audio.buffer]);
  } catch (error) {
    try { await extractor.dispose(); } catch { /* Termination also releases resources. */ }
    self.postMessage({type: 'error', message: error.message});
  }
};
