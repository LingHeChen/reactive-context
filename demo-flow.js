/* An illustrated before/after sequence of the bundled records; not live execution. */
(() => {
  const host = document.querySelector('#change-animation');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let timeline;
  let step = 0;
  const stages = [
    {
      titles: ['01 / 读取三个来源', '02 / 生成模型请求', '03 / 得到三条记录'],
      before: [['尚未解析'], ['尚未编译'], ['尚无结果']],
      after: [['Skill · 最多 3 条', 'Tool · query_database', 'Schema · customers(id, name)'], ['query_database', 'SELECT id, name FROM customers', 'ORDER BY id LIMIT 3;'], ['JSON · 3 条记录', '1 Ada   /   2 Lin   /   3 Mei']],
      labels: ['首次解析', '执行查询'],
      impact: '初始请求：三个来源首次解析，生成 LIMIT 3 的查询，执行后返回 Ada、Lin、Mei。',
    },
    {
      titles: ['01 / 规则收紧', '02 / 查询上限改变', '03 / 结果收缩'],
      before: [['Skill v1 · 最多返回 3 条']],
      after: [['Skill v2 · 最多返回 1 条', '工具与表结构复用'], ['ORDER BY id LIMIT 1;', 'query_database 保持不变'], ['保留 1 Ada', '2 Lin、3 Mei 不再返回']],
      labels: ['重新编译', '只返回一条'],
      impact: '影响：只更新 Skill；SQL 的 LIMIT 从 3 改为 1，结果从三条变成一条，工具与表结构保持不变。',
    },
    {
      titles: ['01 / 替换工具', '02 / 调用名称改变', '03 / 输出格式改变'],
      before: [['Tool v1 · query_database'], ['调用 query_database', 'ORDER BY id LIMIT 1;'], ['JSON', '[{"id":1,"name":"Ada"}]']],
      after: [['Tool v2 · export_csv', '规则与表结构复用'], ['调用 export_csv', 'SQL 保持不变 · LIMIT 1'], ['CSV', 'id,name', '1,Ada']],
      labels: ['更新可用工具', '导出 CSV'],
      impact: '影响：模型可见工具切换为 export_csv，调用工具随之改变；同一条记录从 JSON 变为 CSV，LIMIT 1 继续生效。',
    },
    {
      titles: ['01 / 表结构升级', '02 / SQL 引用同步替换', '03 / CSV 表头更新'],
      before: [['Schema v1', 'customers(id, name)'], ['SELECT id, name', 'FROM customers'], ['id,name', '1,Ada']],
      after: [['Schema v2', 'customer_accounts', '(id, display_name)'], ['SELECT id, display_name', 'FROM customer_accounts', 'ORDER BY id LIMIT 1;'], ['id,display_name', '1,Ada']],
      labels: ['使用新结构', '字段名传递'],
      impact: '影响：customers → customer_accounts，name → display_name；SQL 和 CSV 表头一起更新，导出工具、单条限制和 Ada 记录继续保留。',
    },
  ];
  stages[1].before.push(['ORDER BY id LIMIT 3;'], ['1 Ada   /   2 Lin   /   3 Mei']);
  const esc = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  const lines = (values, x, y, cls) => values.map((value, i) => `<text class="${cls}" x="${x}" y="${y + i * 23}">${esc(value)}</text>`).join('');
  function render(animate = true) {
    timeline?.kill();
    const data = stages[step];
    host.innerHTML = `<svg class="change-svg" viewBox="0 0 1120 305" role="img" aria-labelledby="change-svg-title change-svg-desc"><title id="change-svg-title">${esc(data.impact)}</title><desc id="change-svg-desc">从左到右依次展示来源修改、请求变化和结果变化，上方是修改前，下方是修改后。</desc>${data.titles.map((title, i) => {
      const x = i * 390;
      return `<g><rect class="change-panel" x="${x+1}" y="1" width="338" height="298" rx="10"/><text class="change-title" x="${x+20}" y="32">${title}</text><text class="change-tag" x="${x+20}" y="64">${step===0?'解析前':'修改前'}</text>${lines(data.before[i],x+20,88,'change-before')}<path class="change-divider" d="M${x+20} 143h298"/><g class="change-new" data-phase="${i}"><rect class="change-highlight" x="${x+8}" y="151" width="324" height="139" rx="5"/><text class="change-tag new-tag" x="${x+20}" y="177">${step===0?'解析后':'修改后'}</text>${lines(data.after[i],x+20,206,'change-after')}</g></g>`;
    }).join('')}${[0,1].map(i=>`<g><path class="change-arrow" d="M${i*390+345} 145h38l-6-5m6 5-6 5"/><circle class="change-packet" data-packet="${i}" cx="${i*390+345}" cy="145" r="3"/></g>`).join('')}</svg>`;
    document.querySelector('#change-impact').textContent = data.impact;
    const disabled = reduced.matches || document.documentElement.classList.contains('motion-paused') || document.hidden;
    if (!animate || disabled || !window.gsap) return;
    const panels = host.querySelectorAll('.change-new');
    timeline = gsap.timeline();
    gsap.set(panels, { opacity: 0, y: 7 });
    panels.forEach((panel, i) => {
      timeline.to(panel, { opacity: 1, y: 0, duration: .55, ease: 'power2.out' }, i * 1.05);
      if (i < 2) {
        const packet = host.querySelector(`[data-packet="${i}"]`);
        timeline.fromTo(packet, { opacity: 1, x: 0 }, { x: 38, duration: .5, ease: 'none' }, i * 1.05 + .55)
          .set(packet, { opacity: 0 }, (i+1)*1.05);
      }
    });
  }
  document.addEventListener('context-demo-stage', event => { step = event.detail.index; render(); });
  document.addEventListener('context-motion-change', () => render(false));
  document.addEventListener('visibilitychange', () => { if(document.hidden) render(false); });
  document.querySelector('#replay-change').addEventListener('click', () => render());
  render(false);
})();
