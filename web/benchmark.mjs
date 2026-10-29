const $ = id => document.getElementById(id);
let worker;
$('cancel').onclick = () => { worker?.terminate(); $('run').disabled = false; $('cancel').disabled = true; };
$('run').onclick = async () => {
  $('run').disabled = true; $('cancel').disabled = false; $('result').textContent = '';
  try {
    const context = new AudioContext({sampleRate: 16000});
    let reference;
    try {
      const response = await fetch('/benchmark-reference.wav');
      if (!response.ok) throw Error('Copy the benchmark reference first');
      const decoded = await context.decodeAudioData(await response.arrayBuffer());
      reference = new Float32Array(decoded.length);
      for (let c = 0; c < decoded.numberOfChannels; c++) {
        const channel = decoded.getChannelData(c);
        for (let i = 0; i < channel.length; i++) reference[i] += channel[i] / decoded.numberOfChannels;
      }
    } finally { await context.close(); }
    worker = new Worker(new URL('./benchmark-worker.mjs', import.meta.url), {type: 'module'});
    worker.onmessage = ({data}) => {
      if (data.progress) $('progress').textContent = data.progress;
      if (data.result || data.error) {
        $('result').textContent = data.error ?? JSON.stringify(data.result, null, 2);
        worker.terminate(); $('run').disabled = false; $('cancel').disabled = true;
      }
    };
    worker.onerror = e => { $('result').textContent = e.message; worker.terminate(); $('run').disabled = false; $('cancel').disabled = true; };
    worker.postMessage({reference, budget: $('budget').valueAsNumber, first: $('order').value}, [reference.buffer]);
  } catch (e) { $('result').textContent = e.message; $('run').disabled = false; $('cancel').disabled = true; }
};
