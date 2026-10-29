import {wavBlob} from './math.mjs';
const $ = id => document.getElementById(id);
let worker, cancelRun, cancelled = false, urls = [];
const log = message => { $('log').textContent += message + '\n'; $('log').scrollTop = $('log').scrollHeight; };
$('cancel').onclick = () => { cancelled = true; worker?.terminate(); cancelRun?.(new DOMException('취소됨', 'AbortError')); };
$('start').onclick = async () => {
  const file = $('audio').files[0];
  if (!file) { $('status').textContent = '참조 음성을 선택하세요.'; return; }
  const iterations = $('iterations').valueAsNumber, durationIterations = $('duration').valueAsNumber;
  if (!Number.isInteger(iterations) || iterations < 0 || iterations > 10000 ||
      !Number.isInteger(durationIterations) || durationIterations < 1 || durationIterations > 1000) {
    $('status').textContent = '최적화 횟수를 입력 범위 안의 정수로 설정하세요.'; return;
  }
  $('start').disabled = true; $('cancel').disabled = false;
  $('result').replaceChildren(); $('log').textContent = '';
  urls.forEach(URL.revokeObjectURL); urls = [];
  cancelled = false;
  worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
  let context;
  try {
    context = new AudioContext({sampleRate: 16000});
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (decoded.duration < 0.3 || decoded.duration > 15) throw Error('0.3초 이상 15초 이하의 음성을 선택하세요.');
    if (cancelled) throw new DOMException('취소됨', 'AbortError');
    const reference = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const channel = decoded.getChannelData(c);
      for (let i = 0; i < channel.length; i++) reference[i] += channel[i] / decoded.numberOfChannels;
    }
    await context.close(); context = null;
    if (cancelled) throw new DOMException('취소됨', 'AbortError');
    if (reference.every(v => Math.abs(v) < 1e-7)) throw Error('음성에서 소리를 찾지 못했습니다.');
    const result = await new Promise((resolve, reject) => {
      cancelRun = reject;
      worker.onerror = event => reject(Error(event.message || '추출 작업 오류'));
      worker.onmessage = ({data}) => {
        if (data.type === 'log') { log(data.message); $('status').textContent = data.message; }
        if (data.type === 'error') reject(Error(data.message));
        if (data.type === 'done') resolve(data.result);
        if (data.type === 'step') {
          const e = data.step;
          $('status').textContent = `${e.stage} · ${e.step}회 · 최선 손실 ${(e.bestLoss ?? e.loss).toFixed(5)}`;
        }
      };
      worker.postMessage({type: 'extract', base: new URL($('models').value, location.href).href,
        reference, options: {iterations, durationIterations}}, [reference.buffer]);
    });
    result.style.metadata.source_file = file.name;
    const download = (blob, name, title) => {
      const link = document.createElement('a'); link.className = 'download';
      link.href = URL.createObjectURL(blob); urls.push(link.href); link.download = name; link.textContent = title;
      $('result').append(link); return link.href;
    };
    const name = file.name.replace(/\.[^.]+$/, '');
    download(new Blob([JSON.stringify(result.style)], {type: 'application/json'}), `${name}.json`, '스타일 JSON 저장');
    download(new Blob([JSON.stringify(result.report, null, 2)], {type: 'application/json'}), `${name}.report.json`, '추출 보고서 저장');
    const wavURL = download(wavBlob(result.audio), `${name}.preview.wav`, '미리듣기 WAV 저장');
    const player = document.createElement('audio'); player.controls = true; player.src = wavURL; $('result').append(player);
    $('status').textContent = `완료 · 음색 손실 ${result.report.timbre_loss.toFixed(5)} · 속도 보정: ${result.report.duration.status}`;
    log(JSON.stringify(result.report, null, 2));
  } catch (error) {
    $('status').textContent = error.name === 'AbortError' ? '취소했습니다.' : `오류: ${error.message}`;
    console.error(error);
  } finally {
    try { await context?.close(); worker?.terminate(); worker = null; cancelRun = null; }
    finally { $('start').disabled = false; $('cancel').disabled = true; }
  }
};
