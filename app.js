'use strict';

// Presentation only: all task outputs below come from the bundled Python recording.
// Ambient light lives in fluid.js with a CSS fallback; scroll choreography lives in scroll.js.
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let recording = null;
let currentStep = 0;
let autoDemo = false;
let timer = null;
let toastTimer = null;
let motionEnabled = !reducedMotion.matches;
const controls = $$('.track-tabs button, #replay-toggle');
controls.forEach((button) => { button.disabled = true; });

function toast(message) {
  clearTimeout(toastTimer);
  $('#toast').textContent = message;
  $('#toast').classList.add('visible');
  toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 2500);
}

function showStep(index) {
  if (!recording) return;
  currentStep = (index + recording.stages.length) % recording.stages.length;
  const stage = recording.stages[currentStep];
  const parts = Object.fromEntries(stage.parts.map((part) => [part.part, part]));
  const schema = parts.dbschema.value.tables[0];
  const limit = parts.skill.value.match(/最多返回\s*(\d+)\s*条/);
  $('#skill-value').textContent = limit ? `最多返回 ${limit[1]} 条记录` : parts.skill.value;
  $('#tool-value').textContent = stage.request_frames[0].visible_tools.join(', ');
  $('#schema-value').textContent = `${schema.name}(${schema.columns.join(', ')})`;
  $('#skill-version').textContent = `v${parts.skill.version}`;
  $('#tool-version').textContent = `v${parts.tool.version}`;
  $('#schema-version').textContent = `v${parts.dbschema.version}`;
  $$('.source-row').forEach((row) => {
    const part = parts[row.dataset.source];
    row.classList.toggle('changed', part.change !== 'REUSED');
    row.title = `revision: ${part.revision}\n${part.change} · resolve=${part.resolves}`;
  });
  $('#visible-tool').textContent = stage.output.tool;
  $('#sql-output').textContent = stage.output.sql.replace(' FROM ', '\nFROM ').replace(' ORDER BY ', '\nORDER BY ');
  const result = stage.output.result;
  $('#result-output').textContent = typeof result === 'string'
    ? result.trim()
    : '[\n  ' + result.map((row) => JSON.stringify(row)).join(',\n  ') + '\n]';
  $('#result-format').textContent = typeof result === 'string' ? 'CSV' : 'JSON';
  const changed = stage.parts.find((part) => part.change === 'CHANGED');
  const sourceNames = { skill: 'Skill', tool: '工具目录', dbschema: '数据库 Schema' };
  $('#change-note').textContent = currentStep === 0
    ? '首次解析全部来源'
    : `仅${sourceNames[changed.part]}重新解析，其余复用`;
  $('#step-counter').textContent = `快照 ${currentStep + 1} / 4`;
  $('#step-counter').title = stage.snapshot_id;
  $('#demo-panel').setAttribute('aria-labelledby', `tab-${currentStep}`);
  $$('.track-tabs button').forEach((button, i) => {
    button.setAttribute('aria-selected', String(i === currentStep));
    button.tabIndex = i === currentStep ? 0 : -1;
  });
  document.dispatchEvent(new CustomEvent('context-demo-stage', { detail: { index: currentStep } }));
}

// Scroll and manual controls share the same recorded snapshots.
window.contextDemo = {
  get ready() { return Boolean(recording); },
  setStep(index) { if (index !== currentStep) showStep(index); },
  stop() { setAutoDemo(false); },
};
function selectStep(index) {
  setAutoDemo(false);
  showStep(index);
  document.dispatchEvent(new CustomEvent('context-demo-select', { detail: { index } }));
}
function scheduleNext() {
  clearTimeout(timer);
  if (!autoDemo || document.hidden) return;
  timer = setTimeout(() => {
    if (currentStep === recording.stages.length - 1) {
      setAutoDemo(false);
      toast('四个阶段已展示完毕，可以选择任意阶段查看');
    } else {
      showStep(currentStep + 1);
      scheduleNext();
    }
  }, 4500);
}

function setAutoDemo(enabled) {
  autoDemo = Boolean(enabled && recording);
  $('.replay-label').textContent = window.contextScrollDemo ? '从头演示 ↓' : autoDemo ? '停止演示' : '自动演示';
  $('.play-icon').textContent = autoDemo ? 'Ⅱ' : '↗';
  $('#replay-toggle').setAttribute('aria-label', autoDemo ? '停止自动演示' : '自动演示四个阶段');
  $('#replay-toggle').setAttribute('aria-pressed', String(autoDemo));
  scheduleNext();
}

$('#replay-toggle').addEventListener('click', () => {
  if (window.contextScrollDemo) { window.contextScrollDemo.restart(); return; }
  if (!autoDemo && currentStep === 3) showStep(0);
  setAutoDemo(!autoDemo);
});
$$('.track-tabs button').forEach((button, i) => {
  button.addEventListener('click', () => { selectStep(i); });
  button.addEventListener('keydown', (event) => {
    let next = i;
    if (event.key === 'ArrowRight') next = (i + 1) % 4;
    else if (event.key === 'ArrowLeft') next = (i + 3) % 4;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = 3;
    else return;
    event.preventDefault();
    selectStep(next);
    $(`#tab-${next}`).focus({ preventScroll: true });
  });
});
$('.studio').addEventListener('focusin', (event) => {
  if (!event.target.closest('#replay-toggle')) setAutoDemo(false);
});
new IntersectionObserver((entries) => {
  if (!entries[0].isIntersecting) setAutoDemo(false);
}, { threshold: 0 }).observe($('.studio'));

fetch('./assets/demo-recording.json')
  .then((response) => {
    if (!response.ok) throw new Error('recording unavailable');
    return response.json();
  })
  .then((data) => {
    if (data.stages?.length !== 4 || data.stages.some((stage) => !stage.input_history_unchanged)) {
      throw new Error('recording format invalid');
    }
    recording = data;
    controls.forEach((button) => { button.disabled = false; });
    showStep(0);
    setAutoDemo(false);
    document.dispatchEvent(new Event('context-demo-ready'));
  })
  .catch(() => {
    $('#recording-note').textContent = '执行记录未能加载。请通过本地 HTTP 服务打开页面后刷新；当前静态内容仅作界面预览，交互已禁用。';
    $('#recording-note').style.color = 'var(--gold)';
  });

$('#copy-install').addEventListener('click', async () => {
  const command = $('.install code').textContent;
  try {
    await navigator.clipboard.writeText(command);
    toast('已复制，在源码仓库根目录运行');
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($('.install code'));
    selection.removeAllRanges();
    selection.addRange(range);
    toast('命令已选中，请按 Ctrl/Cmd + C 复制');
  }
});

// Only visible artwork animates. The browser owns frame scheduling and compositing.
const artworkObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => entry.target.classList.toggle('in-view', entry.isIntersecting));
}, { threshold: 0 });
$$('.fluid-art').forEach((artwork) => artworkObserver.observe(artwork));

function setMotion(enabled) {
  motionEnabled = enabled && !reducedMotion.matches;
  document.documentElement.classList.toggle('motion-paused', !motionEnabled);
  const button = $('#motion-toggle');
  button.setAttribute('aria-pressed', String(!motionEnabled));
  button.setAttribute('aria-label', motionEnabled ? '暂停页面动效' : '开启页面动效');
  button.querySelector('span').textContent = motionEnabled ? '暂停动效' : '开启动效';
  button.querySelector('svg').innerHTML = motionEnabled
    ? '<path d="M8 5v14M16 5v14"/>'
    : '<path d="m8 5 11 7-11 7Z"/>';
  document.dispatchEvent(new CustomEvent('context-motion-change', { detail: { enabled: motionEnabled } }));
  button.disabled = reducedMotion.matches;
  if (reducedMotion.matches) {
    button.querySelector('span').textContent = '已减弱动态效果';
    button.setAttribute('aria-label', '遵循系统设置，已减弱动态效果');
  }
}
$('#motion-toggle').addEventListener('click', () => setMotion(!motionEnabled));
reducedMotion.addEventListener('change', (event) => {
  setMotion(!event.matches);
  if (event.matches) setAutoDemo(false);
});
document.addEventListener('visibilitychange', () => {
  document.documentElement.classList.toggle('page-hidden', document.hidden);
  scheduleNext();
});
setMotion(motionEnabled);
document.documentElement.classList.toggle('page-hidden', document.hidden);
