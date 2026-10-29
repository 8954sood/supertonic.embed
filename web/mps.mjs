const $ = id => document.getElementById(id);
let runId, timer;
const headers = {'X-Supertonic-Request': '1'};
async function request(url, options) {
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw Error(data.error ?? `HTTP ${response.status}`);
  return data;
}
async function poll() {
  try {
    const data = await request(`/api/status?id=${runId}`), p = data.progress;
    $('log').textContent = data.log; $('log').scrollTop = $('log').scrollHeight;
    if (p.best_loss != null) {
      $('status').textContent = p.stage === 'duration' ? '말하기 속도 보정 중… 최선 음색은 저장할 수 있습니다.'
        : `음색 ${p.step}회 · 최선 손실 ${p.best_loss.toFixed(5)}`;
      $('metrics').textContent = `목표 ${p.target_loss ?? p.settings?.threshold ?? 0.24} · 초기 ${p.initial_loss.toFixed(5)} · ${p.elapsed_seconds.toFixed(1)}초 · 순전파 ${p.evaluations}회 · 역전파 ${p.backward_calls}회 · 개선 ${p.improvements}회`;
      if (p.steps_since_improvement >= 100) $('metrics').textContent += ' · 100회 이상 개선 없음 (계속 실행 중)';
    } else $('status').textContent = data.cancel_requested ? '취소 중…' : '모델 준비 및 초기 프리셋 비교 중…';
    if (data.cancel_requested && !data.done) $('status').textContent = '취소 중… 최선 결과를 유지합니다.';
    $('downloads').replaceChildren();
    const labels = {'best.json': '현재 최선 저장 (속도 보정 전)', 'style.json': '추출 스타일 저장', 'report.json': '보고서 저장', 'preview.wav': '미리듣기 저장'};
    for (const file of data.files) {
      const a = document.createElement('a'); a.href = `/files/${runId}/${file}`; a.download = file; a.textContent = labels[file]; $('downloads').append(a);
    }
    if (data.done) {
      clearTimeout(timer); $('start').disabled = false; $('cancel').disabled = true; $('threshold').disabled = false;
      $('status').textContent = data.exit_code !== 0 && !data.cancel_requested ? '오류가 발생했습니다. 아래 실행 기록을 확인하세요.'
        : data.cancel_requested || p.status === 'cancelled' ? '취소했습니다. 저장된 최선 스타일은 내려받을 수 있습니다.'
        : p.status === 'threshold reached' ? `목표 도달 · 손실 ${p.best_loss.toFixed(5)}` : `횟수 제한 도달 · 목표 미달 · 손실 ${p.best_loss.toFixed(5)}`;
      if (data.files.includes('preview.wav')) {
        const audio = document.createElement('audio'); audio.controls = true; audio.src = `/files/${runId}/preview.wav`; $('preview').replaceChildren(audio);
      }
    } else timer = setTimeout(poll, 1000);
  } catch (e) {
    $('status').textContent = `상태 확인 실패: ${e.message} · 로컬 서버를 확인하세요.`;
    // A transient polling failure must not abandon an active extraction.
    timer = setTimeout(poll, 3000);
  }
}
$('start').onclick = async () => {
  const file = $('audio').files[0], iterations = $('iterations').valueAsNumber;
  const threshold = $('threshold').valueAsNumber;
  const skipDuration = $('skip').checked;
  const duration = skipDuration ? 100 : $('duration').valueAsNumber;
  const invalid = (message, input) => { $('status').textContent = message; $(input).focus(); };
  if (!file) return invalid('참조 음성의 “파일 선택”을 눌러 음성 파일을 먼저 선택하세요. 새로고침하면 파일을 다시 선택해야 합니다.', 'audio');
  if (file.size === 0) return invalid('선택한 음성 파일이 비어 있습니다. 다른 파일을 선택하세요.', 'audio');
  if (file.size > 32 * 1024 * 1024) return invalid('선택한 음성 파일이 32 MB를 초과합니다. 더 작은 파일을 선택하세요.', 'audio');
  if (!Number.isFinite(threshold) || threshold < 0)
    return invalid('목표 손실은 0 이상의 숫자로 입력하세요. 예: 0.1 또는 0.05', 'threshold');
  if (!Number.isInteger(iterations) || iterations < 0 || iterations > 10000)
    return invalid('음색 최대 갱신 횟수는 0–10000 사이의 정수로 입력하세요. 0은 무제한입니다.', 'iterations');
  if (!Number.isInteger(duration) || duration < 1 || duration > 1000)
    return invalid('속도 보정 단계별 최대 횟수는 1–1000 사이의 정수로 입력하세요.', 'duration');
  $('start').disabled = true; $('cancel').disabled = true; $('threshold').disabled = true; $('preview').replaceChildren();
  $('downloads').replaceChildren(); $('metrics').textContent = ''; $('status').textContent = '이 맥으로 음성 전달 중…';
  try {
    const data = await request(`/api/start?iterations=${iterations}&threshold=${threshold}&duration=${duration}&skip_duration=${skipDuration ? 1 : 0}`,
      {method: 'POST', headers, body: file});
    runId = data.id; $('cancel').disabled = false;
    await poll();
  } catch (e) { $('status').textContent = `오류: ${e.message}`; $('start').disabled = false; $('threshold').disabled = false; }
};
$('skip').onchange = () => { $('duration').disabled = $('skip').checked; };
$('cancel').onclick = async () => {
  $('cancel').disabled = true;
  try { await request(`/api/cancel?id=${runId}`, {method: 'POST', headers}); $('status').textContent = '취소 중… 최선 결과를 유지합니다.'; }
  catch (e) { $('status').textContent = e.message; $('cancel').disabled = false; }
};
