/* 健康顾问培训系统 —— 前端逻辑
 *
 * 两个功能：
 *   1. 知识问答（支持 SSE 流式）
 *   2. 情景演练（对话 + 摘要卡）
 *
 * 无框架、无构建步骤。改完刷新浏览器即可。
 */
'use strict';

const $ = (id) => document.getElementById(id);

/* ================================================================
 * 通用
 * ================================================================ */
// 当前登录态
let AUTH = { token: '', employee_no: '', role: '', role_label: '', is_manager: false };

/**
 * 统一构造请求头（含令牌）。
 *
 * ★ 为什么必须抽出来（2026-10-23 实测 401）：
 *   `askStream()` 原来自己拼了一份 headers，只写了 Content-Type，
 *   **漏了 Authorization** → 勾选「流式输出」就 401。
 *   而非流式走 `api()`，它有令牌，所以问题只在流式上暴露。
 *
 *   同一个东西有两份实现，就一定会漂移一份。这里是第二份，
 *   项目里之前也踩过同类（评阅与护栏各写一套规则 → 规则漂移）。
 *   所以：**任何新增请求路径都必须调这个函数，不许自己拼头。**
 */
function authHeaders(extra = {}) {
  const headers = { 'Content-Type': 'application/json', ...extra };
  if (AUTH.token) headers['Authorization'] = 'Bearer ' + AUTH.token;
  return headers;
}

/**
 * 统一 401 处理：清令牌、回登录页。
 *
 * ★ 抽出来的理由同上：`api()` 有、`askStream()` 原来没有，
 *   结果令牌失效时流式那条只会显示「HTTP 401」，用户不知道自己被登出了。
 *
 * ★★ 必须传 `true`（显示登录框）。
 *   原来传的是 `false`，只清令牌**不显示登录框** ——
 *   注释写「回登录页」但代码没回，用户会卡在一个无法操作的空页面上。
 *   这是「注释与实现不一致」的典型：读注释的人以为已经处理了。
 */
function handleUnauthorized() {
  logout(true);
  throw new Error('登录已失效，请重新登录');
}

async function api(path, options = {}) {
  // ★ 合并调用方自定义头，而不是让 `...options` 整个覆盖掉 ——
  //   原写法 `{ headers: authHeaders(), ...options }` 里，
  //   只要某个调用方传了 `headers`（比如将来要加自定义头），
  //   Authorization 就会被**静默丢掉**，症状又是 401，极难查。
  const { headers: extraHeaders, ...rest } = options;
  const res = await fetch(path, {
    ...rest,
    headers: authHeaders(extraHeaders || {}),
  });

  // ★ 401 统一处理：令牌失效 → 回登录页
  //
  // ★ 例外：登录接口自己的 401 不是「令牌失效」，是**密码错误**。
  //   若也走统一处理，会先把刚输入的工号密码清掉、弹出登录框，
  //   用户看到的是「表单被重置」而不是「密码不对」，体验与语义都错了。
  //   这里直接抛给 doLogin 自己处理。
  if (res.status === 401 && !path.includes('/api/auth/login')) {
    handleUnauthorized();
  }

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail || body.message || msg;
    } catch (_) { /* 响应不是 JSON，用状态码 */ }
    throw new Error(msg);
  }
  return res.json();
}

function setStatus(kind, text) {
  const dot = document.querySelector('#status .dot');
  dot.className = 'dot ' + kind;
  $('statusText').textContent = text;
}

/* ================================================================
 * 登录（P8）
 * ================================================================ */
const TOKEN_KEY = 'jk_token';

async function doLogin() {
  const no = $('loginNo').value.trim();
  const pwd = $('loginPwd').value;
  if (!no || !pwd) { $('loginErr').textContent = '请填写工号和密码'; return; }

  $('loginBtn').disabled = true;
  $('loginBtn').textContent = '登录中…';
  $('loginErr').textContent = '';

  try {
    const r = await api('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ employee_no: no, password: pwd }),
    });
    AUTH = { ...r, is_manager: r.role === 'manager' };
    localStorage.setItem(TOKEN_KEY, r.token);
    applyAuthUI();
    $('loginMask').hidden = true;
    $('loginPwd').value = '';
    bootAfterLogin();
  } catch (e) {
    // 401 会被 api() 转成"登录已失效"，这里要还原成登录失败的语义
    $('loginErr').textContent = '工号或密码不正确';
  } finally {
    $('loginBtn').disabled = false;
    $('loginBtn').textContent = '登录';
  }
}

function logout(showMask = true) {
  AUTH = { token: '', employee_no: '', role: '', role_label: '', is_manager: false };
  localStorage.removeItem(TOKEN_KEY);
  // ★ 换账号必须清 session：否则会带着上一个账号的对话 id 去问，
  //   后端虽按 owner 拒绝、给出新会话，但前端会显示一个没有历史的空对话，
  //   用户会以为"追问怎么不灵了"。
  ASK_SESSION = '';
  localStorage.removeItem(SESSION_KEY);
  if (typeof clearChatList === 'function') clearChatList();
  if (showMask) $('loginMask').hidden = false;
  $('who').hidden = true;
}

/** 按角色显示/隐藏功能入口 */
function applyAuthUI() {
  $('who').hidden = false;
  $('whoName').textContent = AUTH.employee_no;
  $('whoRole').textContent = AUTH.role_label || AUTH.role;

  // 没有权限的页签直接隐藏（后端也会拦，这是体验层）
  // ★ 新页签必须同时加到这里，否则会被隐藏（前端硬编码，后端仍会二次校验）。
  //   月度报告用 `drill` 权限：主管可看所有人，学员只看自己。
  const allowed = AUTH.is_manager
    ? ['ask', 'drill', 'exam_manage', 'daily_write', 'employee_view']
    : ['ask', 'drill', 'daily_write'];

  document.querySelectorAll('.tab').forEach((tab) => {
    const feat = tab.dataset.feature;
    const ok = !feat || allowed.includes(feat);
    tab.hidden = !ok;
  });

  // 修改密码：任何登录用户都可用
  const pwdBtn = $('changePwdBtn');
  if (pwdBtn) pwdBtn.hidden = false;

  // 如果当前停留的页签被隐藏了，切回问答
  const active = document.querySelector('.tab.active');
  if (active && active.hidden) {
    document.querySelector('.tab[data-panel="ask"]')?.click();
  }

  // ★ 日报页：工号与身份都从登录态填充，用户不能改。
  //   （原先是可编辑输入框，导致「不填就保存失败」和「能填别人工号」两个问题）
  if ($('dlTrainee')) $('dlTrainee').value = AUTH.employee_no;
  if ($('dlRoleText')) $('dlRoleText').value = AUTH.role_label || AUTH.role;

  // 主管才显示「只看学员」筛选
  if ($('dlFilterTrainee')) {
    $('dlFilterTrainee').disabled = !AUTH.is_manager;
    $('dlFilterTrainee').placeholder = AUTH.is_manager
      ? '工号，空=全部' : '仅主管可用';
  }
}

/** 登录后恢复会话 */
async function restoreSession() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) { $('loginMask').hidden = false; return; }

  AUTH.token = token;
  try {
    const me = await api('/api/auth/me');
    AUTH = { ...AUTH, ...me };
    applyAuthUI();
    $('loginMask').hidden = true;
    bootAfterLogin();
  } catch (e) {
    // 令牌失效（401 已在 api() 里清了 token）
    $('loginMask').hidden = false;
  }
}

/** 登录后要做的初始化 */
function bootAfterLogin() {
  checkHealth();
  loadCases();
  loadPaperList();
  // ★ 权限相关的东西在登录之后才能判断（见 syncBatchVisibility 注释）
  syncBatchVisibility();
  // 学员工号下拉要带 token 才能拉，所以放登录后
  loadTraineeOptions();
  // 账号管理仅主管可见；数据也只在登录后拉
  if (AUTH.is_manager) loadAccounts();
}

$('loginBtn')?.addEventListener('click', doLogin);
$('loginPwd')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') doLogin();
});
$('loginNo')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('loginPwd').focus();
});
$('logoutBtn')?.addEventListener('click', () => {
  batchStopPoll();          // ★ 登出必须停轮询，否则定时器继续打 401
  logout(true);
});


/* ================================================================
 * 顶栏：健康检查
 * ================================================================ */
async function checkHealth() {
  try {
    const h = await api('/api/health');
    const ready = h.ready_ports.length;
    const llm = h.llm_endpoints.length ? h.llm_endpoints.join('→') : '未配置';
    // ★ 合并后端口 7→10（+sparse/task_queue/checkpoint_saver），
    //   分母写死 7 会显示成「10/7」。按必需端口判断健康，只展示实际数。
    setStatus(ready >= 8 ? 'ok' : 'warn',
              `就绪 ${ready} 个端口　LLM ${llm}　护栏 ${h.guardrail_rules['禁用词规则数']} 条`);
  } catch (e) {
    setStatus('bad', '无法连接后端：' + e.message);
  }
}

/* ================================================================
 * 标签页
 * ================================================================ */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('panel-' + tab.dataset.panel).classList.add('active');
  });
});

/* ================================================================
 * 一、知识问答
 * ================================================================ */
const answerEl = $('answer');
const badgeEl = $('answerBadge');
const metaEl = $('answerMeta');

function renderBadge(refused, annotation) {
  if (refused) {
    badgeEl.className = 'badge danger';
    badgeEl.textContent = '已拒答';
  } else if (annotation && annotation.violation_count > 0) {
    // 部分拦截：有可用内容，但也有违规句被标红
    badgeEl.className = 'badge warn';
    badgeEl.textContent = `部分合规（${annotation.violation_count} 处违规）`;
  } else {
    badgeEl.className = 'badge ok';
    badgeEl.textContent = '正常回答';
  }
}

/**
 * 意图展示文案。
 *
 * ★ 为什么要把「来源」也显示出来：
 *   意图直接决定三路权重（图谱/向量/关键词），判错时用户看到的就是
 *   「答非所问」却不知道为什么。显示来源后，一眼能分清
 *   「规则判的」/「调用方指定的」/「没判出来退开放问答」。
 */
const INTENT_SOURCE_LABEL = {
  rule: '规则识别',
  explicit: '指定',
  llm: '模型识别',
  default: '未识别',
};

function intentHint(meta) {
  const src = INTENT_SOURCE_LABEL[meta.intent_source] || '';
  const w = meta.weights;
  const weightText = Array.isArray(w) && w.length === 3
    ? `（图谱 ${w[0]} / 向量 ${w[1]} / 关键词 ${w[2]}）`
    : '';
  const matched = Array.isArray(meta.intent_matched) && meta.intent_matched.length
    ? ` · 命中「${meta.intent_matched.slice(0, 3).join('、')}」`
    : '';
  return `${src}${weightText}${matched}`;
}

/**
 * 渲染多跳拆解说明。
 *
 * ★ 为什么要把「拆成了几步」显示出来：
 *   多跳改变了回答的**结构**（可能是两段、可能是一段带比价）。
 *   用户看到「关于价格」这种分段时，若不解释就会以为系统在答非所问。
 *   显示依赖关系还能让学员看懂「第二步是在第一步的结果里挑的」——
 *   这本身就是培训内容（先给范围，再谈条件）。
 */
const HOP_KIND_LABEL = {
  simple: '检索',
  price_rank: '比价',
};

function renderHops(resp) {
  const hops = resp && resp.hops;
  if (!Array.isArray(hops) || hops.length <= 1) return;   // 单步不显示

  const rows = hops.map((h) => {
    const kind = HOP_KIND_LABEL[h.kind] || h.kind;
    // 「依赖前一步」是最关键的信息，必须显眼
    const dep = h.depends_on_previous
      ? `<b style="color:#d97706">依赖第 ${h.order} 步结果（候选 ${h.candidate_count} 个）</b>`
      : `独立检索（候选 ${h.candidate_count} 个）`;
    return `<li><span>第 ${h.order + 1} 步 · ${kind}</span>
      <span>${escapeHtml(h.question)}</span>
      <span>${dep}</span></li>`;
  }).join('');

  const box = document.createElement('div');
  box.className = 'hop-box';
  box.innerHTML = `<div class="hop-title">这个问题被拆成 ${hops.length} 步</div>
    <ol class="hop-list">${rows}</ol>`;
  metaEl.parentNode.insertBefore(box, metaEl);
}

function renderMeta(meta) {  if (!meta || !Object.keys(meta).length) { metaEl.innerHTML = ''; return; }
  const parts = [];
  if (meta.intent) parts.push(`意图 <b>${meta.intent}</b> <small>${intentHint(meta)}</small>`);
  if (meta.fused !== undefined) parts.push(`融合 <b>${meta.fused}</b> 条`);
  if (meta.vector_hits !== undefined) {
    parts.push(`向量 ${meta.vector_hits} / 关键词 ${meta.keyword_hits} / 图谱 ${meta.graph_hits}`);
  }
  if (meta.elapsed_ms && meta.elapsed_ms.total !== undefined) {
    parts.push(`耗时 <b>${meta.elapsed_ms.total}</b> ms`);
  }
  if (meta.degraded_paths && meta.degraded_paths.length) {
    parts.push(`<span style="color:#d97706">降级：${meta.degraded_paths.join(',')}</span>`);
  }
  if (meta.llm_called === false) parts.push('未调用 LLM');
  if (meta.partial_block) {
    parts.push(`<span style="color:#d97706">部分拦截 ${meta.removed_count} 处</span>`);
  }
  metaEl.innerHTML = parts.map((p) => `<span>${p}</span>`).join('');
}

/* ================================================================
 * 多轮对话（2026-10-24）
 * ================================================================
 * 改造前：问答区只有**一块** `#answer`，每次提问直接覆盖。
 *        想追问只能重新输入，旧答案被冲掉，而且后端也不知道前文。
 *
 * 改造后：
 *   · 每轮追加一条消息（用户气泡 + 回答块），不覆盖
 *   · 每次请求带 session_id，后端据此接上历史（保留 3 轮）
 *   · 「新对话」按钮清空历史（后端只清 turns、不换 id）
 *
 * ★ 为什么 session_id 存 localStorage 而不是内存：
 *   刷新页面后还能接着上一个对话。但**换账号必须清掉** ——
 *   否则会带着别人的 session_id 去问，后端虽然会按 owner 拒绝，
 *   但前端会显示一个没有历史的"空对话"，让人莫名其妙。
 */
const SESSION_KEY = 'jk_ask_session';
let ASK_SESSION = localStorage.getItem(SESSION_KEY) || '';

/**
 * 确保有会话 ID（合并后新增）。
 *
 * ★ 为什么由前端生成：合并进来的多轮后端契约是「不传 session_id = 单轮」，
 *   它**不会**替调用方新建会话（单轮基线逐字节不变的要求）。
 *   所以第一次提问前必须有 id，否则前端永远拿不到 id，追问永远接不上。
 *   格式按后端校验：`[A-Za-z0-9_-]{8,64}`。
 */
function ensureAskSession() {
  if (!ASK_SESSION) {
    const rand = (crypto.randomUUID && crypto.randomUUID()) ||
      `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    ASK_SESSION = `web-${rand.replace(/-/g, '')}`.slice(0, 48);
    localStorage.setItem(SESSION_KEY, ASK_SESSION);
  }
  return ASK_SESSION;
}

function setAskSession(sid, turns) {
  if (sid) {
    ASK_SESSION = sid;
    localStorage.setItem(SESSION_KEY, sid);
  }
  const hint = $('historyHint');
  if (hint) {
    if (turns > 0) {
      hint.textContent = `已记 ${turns} 轮上下文`;
      hint.hidden = false;
    } else {
      hint.textContent = '';
      hint.hidden = true;
    }
  }
}

/** 清掉对话流（点「新对话」或换账号时）。 */
function clearChatList() {
  const list = $('chatList');
  if (!list) return;
  list.innerHTML = '<div class="chat-empty" id="chatEmpty">提问后这里显示回答</div>';
}

/**
 * 往对话流里追加一轮。
 *
 * Returns:
 *     { bubble, body } 两个节点，供流式时逐字填充 `body.textContent`。
 */
function appendChatTurn(question) {
  const list = $('chatList');
  if (!list) return null;
  // 去掉空态提示
  const empty = $('chatEmpty');
  if (empty) empty.remove();

  const turn = document.createElement('div');
  turn.className = 'chat-turn';

  const q = document.createElement('div');
  q.className = 'chat-q';
  q.textContent = question;

  const a = document.createElement('div');
  a.className = 'chat-a';
  a.textContent = '思考中…';

  turn.appendChild(q);
  turn.appendChild(a);
  list.appendChild(turn);
  turn.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  return { turn, bubble: q, body: a };
}

async function askOnce() {
  const question = $('question').value.trim();
  if (!question) return;

  const body = {
    question,
    product_ids: $('productId').value ? [$('productId').value] : [],
    // ★ 不传 allowed_permissions：检索级别由登录角色在服务端决定
    //   （2026-10-23 修越权时已从 schema 删除该字段，前端再传也是死参数）。
    multi_hop: true,
  };
  // ★ 带上会话，后端才能改写追问（没有就现场生成一个并记住）
  body.session_id = ensureAskSession();

  // 清掉上一轮的诊断信息（meta / badge 是全局单块，只显示最近一次）
  metaEl.innerHTML = '';
  document.querySelectorAll('.hop-box').forEach((el) => el.remove());
  badgeEl.className = 'badge';
  badgeEl.textContent = '';
  $('askBtn').disabled = true;

  const ui = appendChatTurn(question);

  try {
    const useStream = $('useStream').checked;
    if (useStream) {
      await askStream(body, ui);
    } else {
      const r = await api('/api/ask', { method: 'POST', body: JSON.stringify(body) });
      renderTurn(ui, r);
      renderBadge(r.refused, r.annotation);
      renderMeta(r.meta);
      renderHops(r);
      setAskSession(r.session_id, r.history_turns);
    }
  } catch (e) {
    if (ui) {
      ui.body.textContent = '请求失败：' + e.message;
      ui.body.className = 'chat-a error';
    }
    badgeEl.className = 'badge danger';
    badgeEl.textContent = '出错';
  } finally {
    $('askBtn').disabled = false;
    $('question').value = '';
    $('question').focus();
  }
}

/** 把一次完整响应渲染进某一轮的气泡里（含违规标红）。
 *
 *  ★ 这里渲染的是 `annotation.segments`，而**不是** `answer`：
 *    `answer` 已剔除违规句（对外安全），`segments` 才含违规原文。
 *    所以前端能看到标红，但复制 `answer` 拿到的是干净版本。
 */
function renderTurn(ui, r) {
  if (!ui) return;
  const ann = r.annotation;

  if (!ann || !ann.segments || !ann.segments.length) {
    setAnswerHtml(ui.body, r.answer || '');
    ui.body.className = 'chat-a';
    return;
  }

  const parts = [];

  if (ann.violation_count > 0) {
    parts.push(`
      <div class="partial-banner">
        <span class="icon">⚠</span>
        <div>
          这段回答里有 <b>${ann.violation_count}</b> 处违规表述，已<b>标红划掉</b>。
          其余合规内容已保留，可直接参考。
          ${r.refused ? '<br>本次<b>全部内容均违规</b>，因此没有可输出的正文。' : ''}
        </div>
      </div>`);
  }

  ann.segments.forEach((s) => {
    if (s.ok) {
      parts.push(`<span class="seg">${escapeHtml(s.text)}</span>`);
    } else {
      const tags = (s.rule_types || [])
        .map((t) => `<span class="seg-badge">${escapeHtml(t)}</span>`).join('');
      parts.push(`
        <div class="seg bad">
          ${tags}
          <span class="seg-text">${escapeHtml(s.text)}</span>
          <div class="seg-why"><b>为什么不能说：</b>${escapeHtml(s.reason)}</div>
        </div>`);
    }
  });

  ui.body.innerHTML = parts.join('');
  ui.body.className = 'chat-a annotated';
}


/**
 * SSE 流式提问。
 *
 * ★ 必须用 ReadableStream 逐块读，不能用 res.json()——
 *   后者会等整个响应结束，流式就白做了。
 */
async function askStream(body, ui) {
  const res = await fetch('/api/ask/stream', {
    method: 'POST',
    // ★ 用 authHeaders()，不要自己拼 —— 漏了令牌就是 401（实测踩坑）
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  // ★ 401 也要走统一处理，否则用户只看到「HTTP 401」不知道自己被登出
  if (res.status === 401) handleUnauthorized();
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let text = '';
  // ★ 流式也要写进对话流（改造前是写全局单块 #answer）
  const target = ui ? ui.body : answerEl;
  target.textContent = '';
  target.classList.add('streaming');

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔事件
    const parts = buffer.split('\n\n');
    buffer = parts.pop();          // 最后一段可能不完整，留到下次

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') continue;

      let ev;
      try { ev = JSON.parse(data); } catch (_) { continue; }

      if (ev.kind === 'text' && ev.text) {
        text += ev.text;
        setAnswerHtml(target, text);
        target.scrollIntoView({ block: 'nearest' });
      } else if (ev.kind === 'done') {
        // ★ 流式每帧都会带 session_id（同一个值），取到即存
        if (ev.session_id) setAskSession(ev.session_id, ev.history_turns);
        renderBadge(ev.refused);
        renderMeta(ev.meta);
        // 流式被截断：告知学员用非流式可看到「标红 + 原因」
        if (ev.refused) {
          if (!text) {
            target.textContent = '（拒答）' + (ev.refusal_reason || '');
          }
          const tip = document.createElement('div');
          tip.className = 'partial-banner';
          tip.style.marginTop = '12px';
          tip.innerHTML = `<span class="icon">💡</span>
            <div>流式模式下一旦命中违规就会<b>立即截断</b>。
            想看到「哪句话违规、为什么不能说」，请关闭右上角的<b>流式输出</b>再问一次。</div>`;
          target.appendChild(tip);
        }
      }
    }
  }
  target.classList.remove('streaming');
}

/* ---------------- 新对话 ---------------- */
$('newConvBtn')?.addEventListener('click', async () => {
  if (!ASK_SESSION) { clearChatList(); setAskSession('', 0); return; }
  try {
    // ★ 后端只清历史、不换 id（见 ask_session_store 的说明）
    await api('/api/ask', {
      method: 'POST',
      body: JSON.stringify({
        question: '（新对话）',
        session_id: ASK_SESSION,
        new_conversation: true,
        multi_hop: false,
      }),
    });
  } catch (e) {
    // 清历史失败不该阻止前端清屏 —— 但要让用户知道
    console.warn('清空会话失败:', e);
  }
  clearChatList();
  setAskSession(ASK_SESSION, 0);
  metaEl.innerHTML = '';
  badgeEl.className = 'badge';
  badgeEl.textContent = '';
  document.querySelectorAll('.hop-box').forEach((el) => el.remove());
});

$('askBtn').addEventListener('click', askOnce);
$('question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') askOnce();
});

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    $('question').value = chip.dataset.q;
    $('productId').value = chip.dataset.p || '';
    askOnce();
  });
});

/* ================================================================
 * 二、情景演练
 * ================================================================ */
let selectedCase = null;
let currentSession = null;

async function loadCases() {
  try {
    const cases = await api('/api/cases');
    const box = $('caseList');
    box.innerHTML = '';
    $('caseBadge').textContent = `共 ${cases.length} 个案例`;

    cases.forEach((c) => {
      const div = document.createElement('div');
      div.className = 'case-item';
      div.dataset.caseId = c.case_id;
      const red = c.is_red_line ? '<span class="red">红线</span>　' : '';
      div.innerHTML = `
        <h3>${c.case_id} ${escapeHtml(c.title)}</h3>
        <p>${red}要素 ${c.element_count}　违禁规则 ${c.rule_count}</p>`;
      div.addEventListener('click', () => {
        document.querySelectorAll('.case-item').forEach((x) => x.classList.remove('selected'));
        div.classList.add('selected');
        selectedCase = c.case_id;
        $('startBtn').disabled = false;
      });
      box.appendChild(div);
    });
  } catch (e) {
    $('caseList').innerHTML = `<div class="empty-tip">加载失败：${e.message}</div>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

/* ================================================================
 * 回答的轻量 Markdown 渲染
 * ================================================================
 * 模型输出带 `**加粗**`、`* 列表`、`> 引用`，直接 textContent 会原样显示符号。
 *
 * ★ 安全边界：先按行 **转义 HTML**，再套自己的标签 ——
 *   绝不对模型原文做 innerHTML，模型输出属于不可信内容。
 * ★ 只支持这四种：加粗 / 行内代码 / 无序列表 / 引用；标题降级为段落。
 */
function mdInline(escaped) {
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function mdToHtml(text) {
  const lines = String(text || '').split('\n');
  const out = [];
  let inList = false;
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };

  for (const raw of lines) {
    const line = raw.replace(/[ \t]+$/, '');
    if (!line.trim()) { closeList(); continue; }

    const li = /^\s*[*-]\s+(.*)$/.exec(line);
    if (li) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${mdInline(escapeHtml(li[1]))}</li>`);
      continue;
    }
    closeList();

    const bq = /^\s*>\s?(.*)$/.exec(line);
    if (bq) {
      out.push(`<blockquote>${mdInline(escapeHtml(bq[1]))}</blockquote>`);
      continue;
    }
    const h = /^\s*#{1,4}\s+(.*)$/.exec(line);
    if (h) {
      out.push(`<p class="md-h">${mdInline(escapeHtml(h[1]))}</p>`);
      continue;
    }
    out.push(`<p>${mdInline(escapeHtml(line))}</p>`);
  }
  closeList();
  return out.join('');
}

/** 把纯文本答案写进气泡（渲染 Markdown）。 */
function setAnswerHtml(el, text) {
  if (!el) return;
  el.classList.add('md');
  el.innerHTML = mdToHtml(text);
}

function renderChat(state) {
  const box = $('chat');
  box.innerHTML = '';
  state.messages.forEach((m) => {
    const div = document.createElement('div');
    div.className = 'msg ' + m.role;
    const who = m.role === 'customer' ? '客' : '学';
    div.innerHTML = `
      <div class="avatar">${who}</div>
      <div class="bubble">${escapeHtml(m.text)}</div>`;
    box.appendChild(div);
  });
  box.scrollTop = box.scrollHeight;
}

function renderDrillState(state) {
  currentSession = state.session_id;
  $('drillCard').hidden = false;
  $('drillTitle').textContent = `${state.case_id}　${state.case_title}`;
  $('drillProgress').textContent =
    `第 ${state.progress_current}/${state.progress_total} 问　已答 ${state.turn_count} 轮`;

  renderChat(state);

  const finished = state.is_finished;
  $('answerInput').disabled = finished;
  $('submitBtn').disabled = finished;
  $('reviewBtn').disabled = false;

  if (finished) {
    $('answerInput').placeholder = '演练已结束，请点击「结束并评阅」查看摘要卡';
    $('answerInput').value = '';
  }
}

$('startBtn').addEventListener('click', async () => {
  if (!selectedCase) return;
  $('startBtn').disabled = true;
  $('cardResult').hidden = true;
  try {
    const state = await api('/api/drill/start', {
      method: 'POST',
      body: JSON.stringify({
        case_id: selectedCase,
        trainee_id: $('traineeId').value.trim(),
      }),
    });
    renderDrillState(state);
  } catch (e) {
    alert('开始演练失败：' + e.message);
  } finally {
    $('startBtn').disabled = false;
  }
});

$('submitBtn').addEventListener('click', async () => {
  const answer = $('answerInput').value.trim();
  if (!answer || !currentSession) return;

  $('submitBtn').disabled = true;
  try {
    const state = await api('/api/drill/answer', {
      method: 'POST',
      body: JSON.stringify({ session_id: currentSession, answer }),
    });
    $('answerInput').value = '';
    renderDrillState(state);
  } catch (e) {
    alert('提交失败：' + e.message);
  } finally {
    $('submitBtn').disabled = $('answerInput').disabled;
    if (!$('answerInput').disabled) $('answerInput').focus();
  }
});

$('reviewBtn').addEventListener('click', async () => {
  if (!currentSession) return;
  $('reviewBtn').disabled = true;
  try {
    const card = await api('/api/drill/review', {
      method: 'POST',
      body: JSON.stringify({ session_id: currentSession }),
    });
    renderSummaryCard(card);
  } catch (e) {
    alert('评阅失败：' + e.message);
  } finally {
    $('reviewBtn').disabled = false;
  }
});

/* ---------------- 摘要卡 ---------------- */
function renderSummaryCard(card) {
  $('cardResult').hidden = false;

  const badge = $('cardBadge');
  if (card.has_critical_violation) {
    badge.className = 'badge danger';
    badge.textContent = `critical ${card.critical_count} 条`;
  } else if (card.missing_required_count > 0) {
    badge.className = 'badge warn';
    badge.textContent = `缺必答 ${card.missing_required_count} 项`;
  } else {
    badge.className = 'badge ok';
    badge.textContent = '未发现违规';
  }

  const parts = [];

  // 顶部提示（红线案例优先）
  if (card.has_critical_violation) {
    parts.push(`<div class="notice danger">
      命中 ${card.critical_count} 条 critical 违规，属违法宣传风险，必须修改。
    </div>`);
  }

  // 基础信息
  parts.push(`
    <div class="sc-section">
      <h3>基础信息</h3>
      <div class="meta" style="border:none;padding:0;margin:0">
        <span>学员 <b>${escapeHtml(card.trainee_id || '未填')}</b></span>
        <span>案例 <b>${escapeHtml(card.case_id)}</b>　${escapeHtml(card.case_title || '')}</span>
        <span>用时 <b>${card.duration_sec}</b> 秒</span>
      </div>
    </div>`);

  // 风险提示
  parts.push('<div class="sc-section"><h3>风险提示</h3>');
  if (card.risk_items.length === 0) {
    parts.push('<div class="empty-tip">未发现违禁表述。</div>');
  } else {
    card.risk_items.forEach((r) => {
      const cls = r.level === 'critical' ? 'critical' : (r.level === 'high' ? 'high' : '');
      parts.push(`
        <div class="risk ${cls}">
          <div class="head">
            <span class="tag">${escapeHtml(r.rule_type)}</span>
            <span class="matched">「${escapeHtml(r.matched)}」</span>
            <span class="loc">${escapeHtml(r.location)}</span>
          </div>
          <div class="reason">${escapeHtml(r.reason)}</div>
        </div>`);
    });
  }
  parts.push('</div>');

  // 要素覆盖
  parts.push('<div class="sc-section"><h3>要素覆盖</h3>');
  card.element_coverage.forEach((c) => {
    const kind = c.required ? '必答' : '建议';
    parts.push(`
      <div class="el">
        <span class="mark">${c.status_mark}</span>
        <span class="name">${escapeHtml(c.element_id)}　${escapeHtml(c.element_name)}
          ${c.evidence ? `<div class="ev">依据：${escapeHtml(c.evidence)}</div>` : ''}
        </span>
        <span class="kind">${kind}</span>
      </div>`);
  });
  parts.push('</div>');

  // 待人工确认
  parts.push('<div class="sc-section"><h3>待人工确认</h3>');
  if (card.pending_human.length === 0) {
    parts.push('<div class="empty-tip">无</div>');
  } else {
    card.pending_human.forEach((p) => {
      parts.push(`<div class="pending-item">${escapeHtml(p)}</div>`);
    });
  }
  parts.push('</div>');

  // ★ 明确说明不给分
  parts.push(`<div class="notice ok" style="margin-top:6px">
    本摘要卡只提供判断依据，<b>不给分数、不下结论</b>。是否合格由主管判定。
  </div>`);

  $('summaryCard').innerHTML = parts.join('');
  $('cardResult').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ================================================================
 * 启动
 * ================================================================ */
// ★ 启动：先恢复会话，登录成功后才初始化各页数据
restoreSession();
setInterval(() => { if (AUTH.token) checkHealth(); }, 30000);

/* ================================================================
 * 三、试卷（P6）
 * ================================================================ */
let currentPaperNo = null;
let exportCache = { exam: '', answer: '', summary: '' };

/** 解析难度配比 "1:3, 2:9, 3:8" → {"1":3,"2":9,"3":8} */
function parseQuota(text) {
  const out = {};
  (text || '').split(/[,，;；]/).forEach((part) => {
    const m = part.trim().match(/^(\d)\s*[:：=]\s*(\d+)$/);
    if (m) out[m[1]] = parseInt(m[2], 10);
  });
  return out;
}

async function buildExam() {
  const body = {
    total: parseInt($('exTotal').value, 10) || 20,
    stage: $('exStage').value,
    title: $('exTitle').value.trim(),
    difficulty_quota: parseQuota($('exDiff').value),
    trainee_id: $('exTrainee').value.trim(),
    created_by: $('exTrainee').value.trim() || '主管',
    persist: $('exPersist').checked,
  };

  $('exBuildBtn').disabled = true;
  $('exBuildBtn').textContent = '组卷中…';

  try {
    const p = await api('/api/exam/build', { method: 'POST', body: JSON.stringify(body) });
    renderPaper(p);
    loadPaperList();
  } catch (e) {
    // 组卷失败要把原因完整显示（可能是「缺某知识点」）
    alert('组卷失败：\n\n' + e.message);
  } finally {
    $('exBuildBtn').disabled = false;
    $('exBuildBtn').textContent = '生成试卷';
  }
}

function renderPaper(p) {
  currentPaperNo = p.paper_no;
  $('examResult').hidden = false;
  $('examResultTitle').textContent = `${p.paper_no}　${p.title || ''}`;

  const badge = $('examResultBadge');
  badge.className = 'badge ' + (p.warnings.length ? 'warn' : 'ok');
  badge.textContent = `${p.total_count} 题　覆盖 ${p.covered_knowledge_points.length} 个知识点`;

  // 元信息
  const diff = Object.entries(p.difficulty_distribution || {})
    .map(([k, v]) => `${'★'.repeat(+k)}${'☆'.repeat(3 - +k)} ${v}`).join('　');
  $('examMeta').innerHTML = [
    `<span>阶段 <b>${p.stage}</b></span>`,
    `<span>难度 ${diff}</span>`,
    `<span>状态 <b>${p.status}</b></span>`,
    `<span>创建人 ${escapeHtml(p.created_by || '—')}</span>`,
  ].join('');

  // 警告
  $('examWarnings').innerHTML = (p.warnings || []).length
    ? p.warnings.map((w) => `<div class="partial-banner"><span class="icon">⚠</span><div>${escapeHtml(w)}</div></div>`).join('')
    : '';

  // 题目预览（试卷页样式，不含答案）
  const html = [];
  html.push(`<h1>${escapeHtml(p.title || '培训考核试卷')}</h1>`);
  html.push(`<blockquote>卷号 ${p.paper_no}　共 ${p.total_count} 题　姓名 ______　得分 ______</blockquote>`);
  html.push('<hr>');
  p.questions.forEach((q) => {
    const hint = q.question_type === '多选' ? '（多选）' : '（单选）';
    html.push(`<p><strong>${q.seq}. ${escapeHtml(q.stem)}</strong>${hint}</p>`);
    q.options.forEach((opt, i) => {
      html.push(`<p style="margin-left:12px">${String.fromCharCode(65 + i)}. ${escapeHtml(opt)}</p>`);
    });
  });
  $('examPreview').innerHTML = html.join('');

  // 按钮状态
  const isDraft = p.status === 'draft';
  $('exActivateBtn').disabled = !isDraft;
  $('exActivateBtn').textContent = isDraft ? '定稿为生效' : '已定稿';
  $('exExportBtn').disabled = isDraft;
  $('exExportBtn').textContent = isDraft ? '导出（需先定稿）' : '导出';

  $('exportBox').hidden = true;
}

async function activatePaper() {
  if (!currentPaperNo) return;
  if (!confirm('定稿后题目将不可修改，且不能退回草稿。确定？')) return;

  try {
    await api(`/api/exam/papers/${currentPaperNo}/status`, {
      method: 'POST',
      body: JSON.stringify({ paper_no: currentPaperNo, target: 'active' }),
    });
    const p = await api(`/api/exam/papers/${currentPaperNo}`);
    renderPaper(p);
    loadPaperList();
  } catch (e) {
    alert('定稿失败：' + e.message);
  }
}

async function exportPaper() {
  if (!currentPaperNo) return;
  try {
    const r = await api(`/api/exam/papers/${currentPaperNo}/export`, {
      method: 'POST', body: '{}',
    });
    exportCache = {
      exam: r.exam_markdown || '',
      answer: r.answer_markdown || '',
      summary: r.summary_markdown || '',
    };
    $('exportBox').hidden = false;
    showExport('exam');
  } catch (e) {
    alert('导出失败：' + e.message);
  }
}

function showExport(kind) {
  const label = { exam: '试卷页', answer: '答案页', summary: '组卷报告' }[kind];
  $('exportText').textContent = exportCache[kind] || '（空）';
  $('exportBadge').textContent = label;
}

async function loadPaperList() {
  try {
    const list = await api('/api/exam/papers?limit=30');
    const box = $('paperList');
    box.innerHTML = '';

    if (!list.length) {
      box.innerHTML = '<div class="empty-tip">还没有试卷，先在上面组一份。</div>';
      return;
    }

    list.forEach((p) => {
      const div = document.createElement('div');
      div.className = 'paper-item';
      const st = { draft: '草稿', active: '生效', archived: '归档' }[p.status] || p.status;
      div.innerHTML = `
        <h3>${escapeHtml(p.paper_no)}<span class="status-tag ${p.status}">${st}</span></h3>
        <p>${escapeHtml(p.title || '（未命名）')}　${p.total_count} 题　${escapeHtml(p.created_at || '')}</p>`;
      div.addEventListener('click', async () => {
        try {
          const full = await api(`/api/exam/papers/${p.paper_no}`);
          renderPaper(full);
        } catch (e) {
          alert('加载失败：' + e.message);
        }
      });
      box.appendChild(div);
    });
  } catch (e) {
    $('paperList').innerHTML = `<div class="empty-tip">加载失败：${e.message}</div>`;
  }
}

$('exBuildBtn')?.addEventListener('click', buildExam);
$('exRefreshBtn')?.addEventListener('click', loadPaperList);
$('exActivateBtn')?.addEventListener('click', activatePaper);
$('exExportBtn')?.addEventListener('click', exportPaper);

document.querySelectorAll('[data-export]').forEach((btn) => {
  btn.addEventListener('click', () => showExport(btn.dataset.export));
});

$('copyExport')?.addEventListener('click', async () => {
  const text = $('exportText').textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    $('copyExport').textContent = '已复制 ✓';
    setTimeout(() => { $('copyExport').textContent = '复制到剪贴板'; }, 1600);
  } catch (e) {
    alert('复制失败，请手动选中复制');
  }
});


/* ================================================================
 * 四、日报（P7）
 * ================================================================ */
/** 今天 YYYY-MM-DD */
function todayStr() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

$('dlDate').value = todayStr();

/** 当前角色（来自登录态，不从界面读） */
function dlRole() { return AUTH.role || 'trainee'; }
function dlTrainee() { return $('dlTrainee').value.trim(); }

async function loadMySummary() {
  const t = dlTrainee();
  if (!t) { $('dlHint').textContent = '填写工号后可载入当天日报'; return; }

  try {
    // ★ 不再传 requester/role —— 身份由令牌决定
    const list = await api(
      `/api/summaries?start_date=${$('dlDate').value}&end_date=${$('dlDate').value}`);
    const mine = list.find((s) => s.summary_date === $('dlDate').value);

    if (!mine) {
      $('dlContent').value = '';
      $('dlHint').textContent = '今天还没有日报，写点什么吧。';
      setDailyEditable(true);
      $('dailyBadge').className = 'badge';
      $('dailyBadge').textContent = '未创建';
      return;
    }

    $('dlContent').value = mine.content || '';
    if (mine.status === 'submitted') {
      $('dailyBadge').className = 'badge ok';
      $('dailyBadge').textContent = '已提交';
      $('dlHint').textContent = mine.review_note
        ? `主管评语：${mine.review_note}`
        : '已提交，等待主管评语。';
      setDailyEditable(false);
    } else {
      $('dailyBadge').className = 'badge warn';
      $('dailyBadge').textContent = '草稿';
      $('dlHint').textContent = '草稿可以反复修改，提交后不可改。';
      setDailyEditable(true);
    }
  } catch (e) {
    $('dlHint').textContent = '载入失败：' + e.message;
  }
}

function setDailyEditable(editable) {
  $('dlContent').disabled = !editable;
  $('dlSaveBtn').disabled = !editable;
  $('dlSubmitBtn').disabled = !editable;
}

async function saveSummary(submit) {
  const t = dlTrainee();
  if (!t) { alert('请先填写工号'); return; }

  // ★ 不再传 requester/role —— 服务端从令牌取身份
  const body = {
    summary_date: $('dlDate').value,
    content: $('dlContent').value,
    submit,
  };

  $('dlSaveBtn').disabled = true;
  $('dlSubmitBtn').disabled = true;
  try {
    const r = await api('/api/summaries', { method: 'POST', body: JSON.stringify(body) });
    $('dailyBadge').className = 'badge ' + (r.status === 'submitted' ? 'ok' : 'warn');
    $('dailyBadge').textContent = r.status === 'submitted' ? '已提交' : '草稿';
    $('dlHint').textContent = r.status === 'submitted'
      ? '已提交，不可再修改。' : '草稿已保存。';
    setDailyEditable(r.can_edit);
    loadSummaries();
  } catch (e) {
    // 409 = 已提交还想改，要明确告诉用户
    alert(e.message);
    loadMySummary();
  } finally {
    if ($('dlContent').disabled === false) {
      $('dlSaveBtn').disabled = false;
      $('dlSubmitBtn').disabled = false;
    }
  }
}

async function loadSummaries() {
  const role = dlRole();
  const trainee = $('dlFilterTrainee').value.trim();
  const status = $('dlFilterStatus').value;

  // 学员只能看自己：用当前工号，忽略「只看学员」输入
  const effectiveTrainee = role === 'manager' ? trainee : dlTrainee();
  // ★ 不再传 requester/role；主管用 trainee_no 筛选，学员传了也无效
  const params = new URLSearchParams({ limit: '50' });
  if (role === 'manager' && effectiveTrainee) {
    params.set('trainee_no', effectiveTrainee);
  }
  if (status) params.set('status', status);

  try {
    const list = await api('/api/summaries?' + params.toString());
    $('dlListBadge').textContent = `${list.length} 条`;

    const box = $('dlList');
    box.innerHTML = '';

    if (!list.length) {
      box.innerHTML = '<div class="empty-tip">还没有日报。</div>';
      return;
    }

    list.forEach((s) => {
      const row = document.createElement('div');
      row.className = 'summary-row';
      const st = s.status === 'submitted' ? '已提交' : '草稿';
      row.innerHTML = `
        <div class="summary-head">
          <b>${escapeHtml(s.trainee_no)}</b>
          <span>${escapeHtml(s.summary_date)}</span>
          <span class="status-tag ${s.status}">${st}</span>
        </div>
        <div class="summary-body">${escapeHtml(s.content || '（空）')}</div>
        ${s.review_note ? `<div class="summary-review">主管评语：${escapeHtml(s.review_note)}</div>` : ''}`;

      // 主管可写评语
      if (role === 'manager') {
        const btn = document.createElement('button');
        btn.className = 'chip';
        btn.textContent = s.review_note ? '改评语' : '写评语';
        btn.addEventListener('click', () => {
          const note = prompt('评语（留空可清空）', s.review_note || '');
          if (note === null) return;
          writeReview(s.trainee_no, s.summary_date, note);
        });
        const wrap = document.createElement('div');
        wrap.style.marginTop = '8px';
        wrap.appendChild(btn);
        row.appendChild(wrap);
      }

      box.appendChild(row);
    });
  } catch (e) {
    $('dlList').innerHTML = `<div class="empty-tip">加载失败：${e.message}</div>`;
  }
}

async function writeReview(traineeNo, date, note) {
  try {
    await api('/api/summaries/review', {
      method: 'POST',
      body: JSON.stringify({
        trainee_no: traineeNo,
        summary_date: date,
        note,
      }),
    });
    loadSummaries();
  } catch (e) {
    alert('评语保存失败：' + e.message);
  }
}

async function loadMissLogs() {
  try {
    // ★ 不再传 role —— 服务端从令牌判定
    const d = await api('/api/miss-logs?limit=50');
    $('missBadge').textContent =
      `${(d.top || []).length} 个高频问题 / ${(d.recent || []).length} 条最近记录`;

    // 高频问题（按问题内容，不按学员）
    $('missTop').innerHTML = (d.top || []).length
      ? d.top.map((x) => `
          <div class="miss-item">
            <span class="miss-times">${x.times}×</span>
            <span class="miss-query">${escapeHtml(x.query)}</span>
          </div>`).join('')
      : '<div class="empty-tip">暂无。说明目前问的问题都能答上。</div>';

    // 最近记录
    $('missRecent').innerHTML = (d.recent || []).length
      ? d.recent.slice(0, 20).map((x) => `
          <div class="miss-item">
            <span class="miss-query">${escapeHtml(x.query)}</span>
            <span class="miss-time">${escapeHtml(x.created_at || '')}</span>
          </div>`).join('')
      : '<div class="empty-tip">暂无记录</div>';
  } catch (e) {
    $('missBadge').textContent = '加载失败';
    $('missTop').innerHTML = `<div class="empty-tip">${e.message}</div>`;
  }
}

$('dlSaveBtn')?.addEventListener('click', () => saveSummary(false));
$('dlSubmitBtn')?.addEventListener('click', () => {
  if (!confirm('提交后不能修改，确定？')) return;
  saveSummary(true);
});
$('dlRefreshBtn')?.addEventListener('click', () => { loadSummaries(); loadMissLogs(); });
$('dlTrainee')?.addEventListener('change', loadMySummary);
$('dlDate')?.addEventListener('change', loadMySummary);

// 页签切到日报时懒加载
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    if (tab.dataset.panel === 'daily') {
      loadMySummary();
      loadSummaries();
      loadMissLogs();
    }
  });
});

/* ================================================================
 * 五、月度报告（P1/P2/P3 的界面）
 *
 * 三层结构：能力项（第1层）→ 案例（第2层）→ 逐句明细（第3层）。
 *
 * ★ 为什么雷达图手写 SVG 而不是引 ECharts：
 *   本项目前端零依赖（无框架、无构建、无 CDN）。
 *   6 个能力项的数据量很小，手写 SVG 约 80 行就够，
 *   引一个 300KB 的库不划算，且 CDN 挂了整页就白。
 * ================================================================ */
const RP = {
  threadId: '',
  lastSummary: null,
};

// ★ 批量面板状态提前声明（TDZ）：
//   声明位置必须在任何真实使用之前，否则模块顶层执行时可能 ReferenceError。
const BATCH = {
  timer: null,
  period: '',
};

/**
 * 填充「学员工号」下拉。
 *
 * ★ 为什么用工号做 value 而不是姓名：
 *   姓名可能重名，工号唯一。后端也只认工号。
 *
 * ★ 为什么第一项 value=""：
 *   提交时后端逻辑是「空 → 用当前登录人」，
 *   这样「我自己」不需要前端先知道当前工号。
 *
 * ★ 失败时保留一个「我自己」兜底：
 *   接口挂了也不能让下拉变空白，否则用户连自己的报告都生成不了。
 */
async function loadTraineeOptions() {
  const sel = $('rpTrainee');
  if (!sel) return;

  const keep = sel.value;          // 刷新时保留当前选择
  const rows = await (async () => {
    try {
      return await api('/api/employees?limit=500');
    } catch (e) {
      return null;
    }
  })();

  const opts = ['<option value="">我自己</option>'];
  if (Array.isArray(rows)) {
    // 主管能看到全部；学员只会拿到自己 —— 这里不区分，按返回如实列
    rows
      .filter((e) => e.employee_no)
      .forEach((e) => {
        const name = e.name ? '（' + e.name + '）' : '';
        const off = e.active === false ? ' · 已停用' : '';
        opts.push(
          `<option value="${escapeHtml(e.employee_no)}">` +
          `${escapeHtml(e.employee_no)}${escapeHtml(name)}${escapeHtml(off)}` +
          `</option>`);
      });
  } else {
    // 接口不可用：至少保留「我自己」
    opts.push('<option value="" disabled>（员工列表加载失败）</option>');
  }
  sel.innerHTML = opts.join('');

  // 还原选择（若原来选的工号还在列表里）
  if (keep && [...sel.options].some((o) => o.value === keep)) {
    sel.value = keep;
  }
}

/** 当前周期的 YYYY-MM。 */
function currentPeriod() {
  const d = new Date();
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0');
}

/** 周期 → since/until（右开区间，避免月末边界算错）。 */
function periodRange(period) {
  if (!period) return { since: '', until: '' };
  const [y, m] = period.split('-').map(Number);
  const since = `${y}-${String(m).padStart(2, '0')}-01`;
  const ny = m === 12 ? y + 1 : y;
  const nm = m === 12 ? 1 : m + 1;
  const until = `${ny}-${String(nm).padStart(2, '0')}-01`;
  return { since, until };
}

/** 等级 → 徽标配色。 */
function levelClass(level) {
  // ★ 没有 `good` 了（2026-10-31 去掉「达标」档，见后端 THRESHOLD_WEAK 注释）：
  //   这套评分只回答「有没有问题」，不回答「是否优秀」。
  //
  // ★ `fair` 从 warn 改成 ok：它的语义是「未见明显问题」，
  //   用橙色会让人误以为"有问题但说不清"，反而误导。
  return {
    fair: 'ok', weak: 'danger',
    red_line: 'danger', insufficient: '', not_trained: '',
  }[level] || '';
}

/* ---------------- 三层渲染：能力项（雷达 + 列表） ---------------- */
function renderRadar(caps) {
  const box = $('rpRadar');
  if (!box) return;
  const items = (caps || []).filter((c) => c.score !== null);
  if (items.length < 3) {
    // 少于 3 项画不出雷达，退回条形（诚实降级，不硬画个三角形）
    box.innerHTML = '';
    return;
  }

  const W = 320, H = 260, cx = W / 2, cy = H / 2 + 6, R = 92;
  const n = items.length;
  const step = (Math.PI * 2) / n;

  // 网格（4 圈）
  let grid = '';
  for (let ring = 1; ring <= 4; ring++) {
    const r = (R * ring) / 4;
    const pts = items.map((_, i) => {
      const a = -Math.PI / 2 + step * i;
      return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
    }).join(' ');
    grid += `<polygon points="${pts}" fill="none" stroke="var(--border)" stroke-width="1"/>`;
  }
  // 轴线
  items.forEach((_, i) => {
    const a = -Math.PI / 2 + step * i;
    grid += `<line x1="${cx}" y1="${cy}" x2="${cx + R * Math.cos(a)}"
              y2="${cy + R * Math.sin(a)}" stroke="var(--border)" stroke-width="1"/>`;
  });

  // 数据多边形（分数 0–100）
  const dataPts = items.map((c, i) => {
    const a = -Math.PI / 2 + step * i;
    const r = (R * Math.max(0, Math.min(100, c.score || 0))) / 100;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(' ');

  // 标签
  let labels = '';
  items.forEach((c, i) => {
    const a = -Math.PI / 2 + step * i;
    const lx = cx + (R + 22) * Math.cos(a);
    const ly = cy + (R + 22) * Math.sin(a);
    const anchor = Math.abs(Math.cos(a)) < 0.3 ? 'middle'
      : (Math.cos(a) > 0 ? 'start' : 'end');
    labels += `<text x="${lx}" y="${ly}" text-anchor="${anchor}"
                dominant-baseline="middle" font-size="11" fill="var(--text-2)">
                ${escapeHtml(c.name)}</text>`;
  });

  box.innerHTML = `
    <div class="card-head" style="margin-top:4px">
      <h2>能力雷达</h2>
      <span class="badge">${items.length} 项可评</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:420px;margin:0 auto;display:block">
      ${grid}
      <polygon points="${dataPts}" fill="rgba(13,148,136,.18)"
               stroke="var(--primary)" stroke-width="2"/>
      ${labels}
    </svg>`;
}

function renderCapabilities(caps, weak) {
  const box = $('rpCaps');
  if (!box) return;
  if (!caps || !caps.length) { box.innerHTML = ''; return; }

  const weakSet = new Set(weak || []);
  const rows = caps.map((c) => {
    const scoreText = c.score === null ? c.level_label : `${c.score} 分`;
    const isWeak = weakSet.has(c.key);
    const cls = c.score === null ? 'badge' : `badge ${levelClass(c.level)}`;
    return `
      <div class="rp-cap-row" data-cap="${escapeHtml(c.key)}">
        <div class="rp-cap-name">${escapeHtml(c.name)}
          ${isWeak ? '<span class="badge danger" style="margin-left:6px">薄弱</span>' : ''}
        </div>
        <div class="rp-cap-score">
          <span class="${cls}">${escapeHtml(scoreText)}</span>
          <span class="rp-cap-samples">${c.samples} 次演练</span>
        </div>
        <div class="rp-cap-note">${escapeHtml(c.note || '')}</div>
      </div>`;
  }).join('');

  box.innerHTML = `
    <div class="card-head" style="margin-top:16px"><h2>能力项明细</h2></div>
    ${rows}`;
}

/* ---------------- 第 2 层：案例，可展开第 3 层 ---------------- */
function renderCases(cases) {
  const box = $('rpCases');
  if (!box) return;
  if (!cases || !cases.length) { box.innerHTML = ''; return; }

  const rows = cases.map((c, idx) => {
    const riskList = (c.risks || []).map((r) => `
      <div class="rp-risk rp-risk-${escapeHtml(r.level)}">
        <span class="rp-risk-loc">${escapeHtml(r.location)}</span>
        <span class="rp-risk-text">「${escapeHtml(r.matched)}」</span>
        <span class="rp-risk-type">${escapeHtml(r.rule_type)}</span>
        <div class="rp-risk-reason">${escapeHtml(r.reason || '')}</div>
      </div>`).join('') || '<div class="empty-tip">无逐句违规</div>';

    const head = c.risk_count > 0
      ? `<span class="badge danger">${c.risk_count} 条违规</span>`
      : '<span class="badge ok">无违规</span>';

    return `
      <div class="rp-case" data-idx="${idx}">
        <div class="rp-case-head" onclick="rpToggleCase(${idx})">
          <span class="rp-case-id">${escapeHtml(c.case_id)}</span>
          <span class="rp-case-title">${escapeHtml(c.case_title)}</span>
          ${head}
          <span class="rp-case-score">${c.score} 分</span>
          <span class="rp-case-arrow">▾</span>
        </div>
        <div class="rp-case-body" id="rpCaseBody${idx}" hidden>
          <div class="rp-case-note">${escapeHtml(c.score_note || '')}
            · 必答要素 ${c.element_hit}/${c.element_total}</div>
          ${riskList}
        </div>
      </div>`;
  }).join('');

  box.innerHTML = `
    <div class="card-head" style="margin-top:16px">
      <h2>案例明细</h2>
      <span class="badge">点击展开逐句</span>
    </div>
    ${rows}`;
}

function rpToggleCase(idx) {
  const body = $('rpCaseBody' + idx);
  if (!body) return;
  body.hidden = !body.hidden;
  const head = body.parentElement.querySelector('.rp-case-arrow');
  if (head) head.textContent = body.hidden ? '▾' : '▴';
}

/* ---------------- 概览 / 建议 ---------------- */
function renderOverview(data) {
  const box = $('rpOverview');
  if (!box) return;
  const ov = data.overview || {};
  const weak = data.weak_capabilities || [];
  box.innerHTML = `
    <div class="rp-overview">
      <div class="rp-stat"><b>${ov.total_drills || 0}</b><span>演练次数</span></div>
      <div class="rp-stat"><b>${ov.distinct_cases || 0}</b><span>覆盖案例</span></div>
      <div class="rp-stat"><b>${Math.round((ov.total_duration_sec || 0) / 60)}</b><span>总时长(分)</span></div>
      <div class="rp-stat ${ov.total_critical ? 'danger' : ''}">
        <b>${ov.total_critical || 0}</b><span>红线违规</span></div>
    </div>
    ${data.degraded ? `<div class="partial-banner warn-box">
      <span class="icon">⚠</span><div>${escapeHtml(data.degrade_reason || '')}</div></div>` : ''}
    ${weak.length ? '' : '<div class="empty-tip">本月无薄弱项</div>'}`;
}

function renderSuggestion(sug) {
  const box = $('rpSuggestion');
  if (!box) return;
  if (!sug) { box.innerHTML = ''; return; }

  let body;
  if (sug.skipped) {
    body = `<div class="empty-tip">${escapeHtml(sug.reason || sug.text || '本月无需生成建议')}</div>`;
  } else if (!sug.passed) {
    body = `<div class="partial-banner danger-box"><span class="icon">✗</span>
      <div>建议未通过合规校验：${escapeHtml(sug.reason || '')}</div></div>
      <div class="rp-sugg-text">${escapeHtml(sug.text || '')}</div>`;
  } else {
    body = `<div class="rp-sugg-text">${escapeHtml(sug.text || '')}</div>`;
  }

  box.innerHTML = `
    <div class="card-head" style="margin-top:16px">
      <h2>改进建议</h2>
      ${sug.retries ? `<span class="badge warn">重写 ${sug.retries} 次</span>` : ''}
    </div>
    ${body}`;
}

/** 渲染整份报告（定稿快照或快速查看结果）。 */
function renderReport(data) {
  RP.lastSummary = data;
  $('rpResultCard').hidden = false;
  const badge = $('rpResultBadge');
  badge.className = 'badge ok';
  badge.textContent = '已定稿';

  renderOverview(data);
  renderRadar(data.capabilities || []);
  renderCapabilities(data.capabilities || [], data.weak_capabilities || []);
  renderCases(data.cases || []);
  renderSuggestion(data.suggestion || null);
  if (data.disclaimer) {
    $('rpHint').textContent = data.disclaimer;
  }
}

/* ---------------- 待复核 ---------------- */
function renderPending(pending, threadId) {
  const card = $('rpPendingCard');
  if (!card) return;
  card.hidden = false;
  $('rpPendingBadge').textContent =
    pending.rewrites ? `已重写 ${pending.rewrites} 次` : '待复核';

  const sug = pending.suggestion || '';
  // ★ 用 weak_caps（带中文名）而不是 weak（英文 key）：
  //   原来直接 join(weak)，主管看到的是一串 C_contraindication 这种标识符，
  //   等于没告诉他哪里弱。
  const weakNames = (pending.weak_caps || []).map((c) => c.name || c.key);
  const weak = weakNames.length
    ? weakNames.join('、')
    : ((pending.weak || []).join('、') || '无');
  $('rpPendingBody').innerHTML = `
    <div class="rp-overview">
      <div class="rp-stat"><b>${pending.total_drills || 0}</b><span>演练次数</span></div>
      <div class="rp-stat rp-stat-text"><b>${escapeHtml(weak)}</b><span>薄弱能力项</span></div>
      <div class="rp-stat ${pending.suggestion_passed ? '' : 'danger'}">
        <b>${pending.suggestion_passed ? '通过' : '未通过'}</b><span>建议校验</span></div>
    </div>
    ${pending.suggestion_note ? `<div class="partial-banner danger-box">
      <span class="icon">✗</span><div>${escapeHtml(pending.suggestion_note)}</div></div>` : ''}
    <div class="rp-sugg-text" style="margin-top:10px">${escapeHtml(sug || '（无建议）')}</div>`;

  $('rpResultCard').hidden = true;
  RP.threadId = threadId || '';
}

/* ---------------- 事件绑定 ---------------- */
document.addEventListener('DOMContentLoaded', () => {
  if (!$('rpGenBtn')) return;

  $('rpPeriod').value = currentPeriod();

  // 生成（进复核流程）
  $('rpGenBtn').addEventListener('click', async () => {
    const trainee = $('rpTrainee').value.trim();
    const period = $('rpPeriod').value || currentPeriod();
    const { since, until } = periodRange(period);
    $('rpHint').textContent = '正在生成（会走完整图，约 5–15 秒）…';
    $('rpGenBtn').disabled = true;
    try {
      const d = await api('/api/reports/monthly/generate', {
        method: 'POST',
        body: JSON.stringify({ trainee_id: trainee, period, since, until }),
      });
      if (d.status === 'pending_review') {
        renderPending(d.pending || {}, d.thread_id);
        $('rpHint').textContent = '已生成，请在下方复核后定稿。';
      } else {
        $('rpHint').textContent = '生成完成：' + (d.status || '');
      }
    } catch (e) {
      $('rpHint').textContent = '生成失败：' + e.message;
    } finally {
      $('rpGenBtn').disabled = false;
    }
  });

  // 快速查看（同步版，不复核）
  $('rpQuickBtn').addEventListener('click', async () => {
    const trainee = $('rpTrainee').value.trim();
    const period = $('rpPeriod').value || currentPeriod();
    const { since, until } = periodRange(period);
    $('rpHint').textContent = '正在查询…';
    $('rpQuickBtn').disabled = true;
    try {
      const qs = new URLSearchParams({ trainee_id: trainee, since, until });
      const d = await api('/api/reports/monthly?' + qs);
      $('rpPendingCard').hidden = true;
      renderReport(d);
      $('rpResultBadge').textContent = '快速查看（未复核）';
      $('rpResultBadge').className = 'badge warn';
      $('rpHint').textContent = d.disclaimer || '';
    } catch (e) {
      $('rpHint').textContent = '查询失败：' + e.message;
    } finally {
      $('rpQuickBtn').disabled = false;
    }
  });

  // ---- 批量（放一起初始化，避免两个 DOMContentLoaded 抢执行顺序）----
  const bPeriod = $('rpBatchPeriod');
  if (bPeriod) {
    bPeriod.value = currentPeriod();
    $('rpBatchRunBtn').addEventListener('click', batchRun);
    $('rpBatchRefreshBtn').addEventListener('click', () => {
      BATCH.period = bPeriod.value || currentPeriod();
      batchFetchProgress();
    });
  }

  // 同意
  $('rpApproveBtn').addEventListener('click', () => reviewReport('approve'));
  // 驳回
  $('rpRejectBtn').addEventListener('click', () => reviewReport('reject'));

  async function reviewReport(decision) {
    if (!RP.threadId) {
      $('rpReviewHint').textContent = '没有待复核的报告';
      return;
    }
    const note = $('rpReviewNote').value.trim();
    $('rpReviewHint').textContent =
      decision === 'approve' ? '正在定稿…' : '正在重新生成建议…';
    $('rpApproveBtn').disabled = true;
    $('rpRejectBtn').disabled = true;
    try {
      const d = await api(
        `/api/reports/monthly/${encodeURIComponent(RP.threadId)}/review`, {
          method: 'POST',
          body: JSON.stringify({ decision, note }),
        });
      if (d.status === 'approved') {
        $('rpPendingCard').hidden = true;
        renderReport(d.report || {});
        $('rpReviewHint').textContent = '已定稿。';
      } else {
        renderPending(d.pending || {}, d.thread_id || RP.threadId);
        $('rpReviewHint').textContent =
          decision === 'reject' ? '已驳回并重新生成，请再次复核。' : '';
      }
    } catch (e) {
      $('rpReviewHint').textContent = '复核失败：' + e.message;
    } finally {
      $('rpApproveBtn').disabled = false;
      $('rpRejectBtn').disabled = false;
    }
  }
});

/* ================================================================
 * 六、批量生成（P3 队列的界面）
 *
 * ★ 只有主管可见（隐藏靠 AUTH.is_manager，后端也会拦 403）。
 *
 * ★★ 轮询必须能停，不能叠加：
 *   这是异步任务界面最容易出的坑 ——
 *   点一次「刷新」起一个 setInterval，再点一次又起一个，
 *   10 次之后有 10 个定时器同时在打接口。
 *   所以这里全程只持有一个 `BATCH.timer`，起新的之前先停旧的，
 *   页面切走 / 登出 / 任务跑完都要停。
 * ================================================================ */
/** 起轮询（先停旧的，避免叠加）。 */
function batchStartPoll() {
  batchStopPoll();
  BATCH.timer = setInterval(() => {
    // 页面不可见时跳过这一拍（省接口、也避免后台堆积）
    if (document.hidden) return;
    batchFetchProgress({ silent: true });
  }, 3000);
}

function batchStopPoll() {
  if (BATCH.timer) {
    clearInterval(BATCH.timer);
    BATCH.timer = null;
  }
}

/** 进度条 + 数字。 */
function batchRenderProgress(p) {
  const box = $('rpBatchProgress');
  if (!box) return;
  const total = p.total || 0;
  const pct = total ? Math.min(100, p.percent || 0) : 0;

  box.innerHTML = `
    <div class="rp-progress-wrap">
      <div class="rp-progress-bar">
        <div class="rp-progress-fill" style="width:${pct}%"></div>
      </div>
      <div class="rp-progress-text">
        <b>${p.generated || 0}/${total}</b> 已生成（${pct}%）
        ${p.queued ? `<span class="rp-tag">排队 ${p.queued}</span>` : ''}
        ${p.running ? `<span class="rp-tag warn">运行 ${p.running}</span>` : ''}
        ${p.pending_review ? `<span class="rp-tag ok">待复核 ${p.pending_review}</span>` : ''}
        ${p.done ? `<span class="rp-tag ok">已定稿 ${p.done}</span>` : ''}
        ${p.failed ? `<span class="rp-tag danger">失败 ${p.failed}</span>` : ''}
        ${p.dead ? `<span class="rp-tag danger">已放弃 ${p.dead}</span>` : ''}
      </div>
    </div>`;
}

/** 进度是否「跑完」（用于决定要不要停轮询）。 */
function batchIsSettled(p) {
  if (!p.total) return false;
  return (p.queued || 0) === 0 && (p.running || 0) === 0;
}

/** 拉进度。 */
async function batchFetchProgress(opts = {}) {
  const period = BATCH.period || $('rpBatchPeriod').value || currentPeriod();
  BATCH.period = period;
  try {
    const qs = new URLSearchParams({ period });
    const p = await api('/api/reports/monthly/batch/progress?' + qs);
    batchRenderProgress(p);

    const badge = $('rpBatchBadge');
    if (badge) {
      if ((p.dead || 0) > 0) {
        badge.className = 'badge danger';
        badge.textContent = `${p.dead} 个需人工处理`;
      } else if (batchIsSettled(p) && p.total) {
        badge.className = 'badge ok';
        badge.textContent = '已生成完';
      } else {
        badge.className = 'badge';
        badge.textContent = '进行中';
      }
    }

    // 明细（逐人状态）
    batchFetchTasks(period);

    // ★ 跑完就停轮询，别让它空转
    if (batchIsSettled(p) && !opts.keepPolling) {
      batchStopPoll();
      $('rpBatchHint').textContent =
        `已生成完：${p.generated}/${p.total}。请到上方「待复核」逐个定稿。`;
    }
    return p;
  } catch (e) {
    if (!opts.silent) $('rpBatchHint').textContent = '读取进度失败：' + e.message;
    // ★ 出错要停轮询，否则会不停打失败的接口
    batchStopPoll();
    return null;
  }
}

/**
 * 拉取并渲染「逐人状态」明细。
 *
 * ★ 进度条只回答「跑到哪了」，不回答「谁卡住了」。
 *   失败 / 放弃的任务必须能定位到人，否则主管只能重新全跑一遍。
 */
async function batchFetchTasks(period) {
  const box = $('rpBatchList');
  if (!box) return;
  let d = null;
  try {
    const qs = new URLSearchParams({ period });
    d = await api('/api/reports/monthly/batch/tasks?' + qs);
  } catch (e) {
    return;                    // 明细拉不到不影响进度显示，静默
  }
  const items = (d && d.items) || [];
  if (!items.length) { box.innerHTML = ''; return; }

  // 按状态排序：异常在前，正常在后（主管先看有问题的）
  const order = { dead: 0, failed: 1, running: 2, queued: 3,
                  pending_review: 4, done: 5 };
  const sorted = items.slice().sort(
    (a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9));

  const cls = {
    dead: 'danger', failed: 'danger', running: 'warn',
    queued: '', pending_review: 'ok', done: 'ok',
  };

  const rows = sorted.map((it) => `
    <div class="rp-task-row">
      <span class="rp-task-no">${escapeHtml(it.trainee_no)}</span>
      <span class="rp-task-name">${escapeHtml(it.name || '')}</span>
      <span class="rp-tag ${cls[it.status] || ''}">${escapeHtml(it.status_label || it.status)}</span>
      ${it.attempts ? `<span class="rp-task-attempts">重试 ${it.attempts} 次</span>` : ''}
      ${it.last_error ? `<span class="rp-task-err" title="${escapeHtml(it.last_error)}">${escapeHtml(it.last_error.slice(0, 30))}…</span>` : ''}
    </div>`).join('');

  box.innerHTML = `
    <div class="card-head" style="margin-top:16px">
      <h3>逐人状态</h3>
      <span class="badge">${items.length} 人</span>
    </div>
    <div class="rp-task-list">${rows}</div>`;
}

/** 触发一次批量投递。 */
async function batchRun() {
  const period = $('rpBatchPeriod').value || currentPeriod();
  BATCH.period = period;
  const btn = $('rpBatchRunBtn');
  btn.disabled = true;
  $('rpBatchHint').textContent = '正在投递任务…';
  try {
    const { since, until } = periodRange(period);
    const d = await api('/api/reports/monthly/batch', {
      method: 'POST',
      // ★ trainee_ids 传空数组：后端会取全部在职员工
      body: JSON.stringify({ period, since, until, trainee_ids: [] }),
    });
    const enq = d.enqueued_now || 0;
    const skip = d.skipped || 0;
    $('rpBatchHint').textContent =
      `已投递 ${enq} 个任务` + (skip ? `，跳过 ${skip} 个（已生成/处理中）` : '');
    batchRenderProgress(d);
    batchStartPoll();
  } catch (e) {
    $('rpBatchHint').textContent = '投递失败：' + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ---------------- 权限（由 bootAfterLogin 调用） ---------------- */
/**
 * 按角色显示/隐藏批量卡。
 *
 * ★★ 为什么不能放在 `DOMContentLoaded` 里（实测会出错）：
 *   `AUTH.is_manager` 要等登录接口返回才赋值，
 *   而 `DOMContentLoaded` 在页面加载时就跑完了 ——
 *   那一刻 `AUTH.is_manager` 还是初始的 `false`，
 *   结果**主管的批量卡也被隐藏了**，功能等于不存在。
 *
 *   所以必须在登录成功后（`bootAfterLogin`）再判断。
 */
function syncBatchVisibility() {
  const card = $('rpBatchCard');
  if (card) card.hidden = !AUTH.is_manager;
  if (!AUTH.is_manager) batchStopPoll();   // 非主管不留定时器
}

// ★ 登出 / 切页签时停轮询（否则定时器会在后台一直打接口）
document.addEventListener('visibilitychange', () => {
  if (document.hidden) batchStopPoll();
});

/* ================================================================
 * 六、账号管理（主管）+ 修改密码
 *
 * 导入只建档案；账号由主管在「账号」页显式开通，避免默认密码窗口期。
 * 密码输入走 type=password 对话框，不用 prompt() 明文回显。
 * ================================================================ */

/** 密码对话框：返回 Promise<string|null>，取消返回 null。 */
function askPassword(title) {
  return new Promise((resolve) => {
    const mask = $('pwdMask');
    const input = $('pwdInput');
    const err = $('pwdErr');
    $('pwdTitle').textContent = title;
    input.value = '';
    err.textContent = '';
    mask.hidden = false;
    input.focus();

    const done = (value) => {
      mask.hidden = true;
      $('pwdOk').removeEventListener('click', onOk);
      $('pwdCancel').removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      resolve(value);
    };
    const onOk = () => {
      const v = input.value;
      if (v.length < 6) { err.textContent = '密码至少 6 位'; return; }
      done(v);
    };
    const onCancel = () => done(null);
    const onKey = (e) => { if (e.key === 'Enter') onOk(); };
    $('pwdOk').addEventListener('click', onOk);
    $('pwdCancel').addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
  });
}

let ACC_LOADING = false;

async function loadAccounts() {
  const body = $('accBody');
  const hint = $('accHint');
  if (!body || ACC_LOADING) return;
  ACC_LOADING = true;
  hint.textContent = '加载中…';
  try {
    const [employees, accounts] = await Promise.all([
      api('/api/employees?active_only=false&limit=500'),
      api('/api/accounts?limit=1000'),
    ]);
    const accByNo = new Map(accounts.map((a) => [a.employee_no, a]));
    const fragment = document.createDocumentFragment();
    if (!employees.length) {
      hint.textContent = '还没有员工档案，请先导入员工。';
      body.replaceChildren(fragment);
      return;
    }
    hint.textContent = `共 ${employees.length} 人，${accounts.length} 个账号`;
    employees.forEach((e) => {
      const a = accByNo.get(e.employee_no);
      const tr = document.createElement('tr');
      const roleText = a ? (a.role === 'manager' ? '主管' : '学员') : '—';
      const accState = a ? (a.active ? '启用' : '停用') : '未开通';
      const empState = e.active === false ? '停用' : '在职';
      tr.innerHTML = `
        <td>${escapeHtml(e.employee_no)}</td>
        <td>${escapeHtml(e.name || '')}</td>
        <td>${a ? '已开通' : '<span class="acc-state">未开通</span>'}</td>
        <td>${escapeHtml(roleText)}</td>
        <td>${escapeHtml(accState)}</td>
        <td>${escapeHtml(empState)}</td>
        <td></td>`;
      const ops = tr.lastElementChild;
      const addBtn = (label, fn, cls = 'btn ghost mini-btn') => {
        const b = document.createElement('button');
        b.className = cls;
        b.textContent = label;
        b.addEventListener('click', fn);
        ops.appendChild(b);
      };
      if (!a) {
        addBtn('开通', () => createAccount(e.employee_no), 'btn primary mini-btn');
      } else {
        addBtn('重置密码', () => resetPassword(e.employee_no));
        addBtn(a.active ? '停用账号' : '启用账号',
               () => toggleAccount(e.employee_no, !a.active));
      }
      addBtn(e.active === false ? '启用档案' : '停用档案',
             () => toggleEmployee(e.employee_no, e.active === false));
      fragment.appendChild(tr);
    });
    body.replaceChildren(fragment);
  } catch (e) {
    hint.textContent = '加载失败：' + e.message;
  } finally {
    ACC_LOADING = false;
  }
}

async function createAccount(employeeNo) {
  const pwd = await askPassword(`为 ${employeeNo} 设置初始密码`);
  if (pwd === null) return;
  try {
    await api('/api/accounts/create', {
      method: 'POST',
      body: JSON.stringify({ employee_no: employeeNo, role: 'trainee', password: pwd }),
    });
    await loadAccounts();
  } catch (e) { alert('开通失败：' + e.message); }
}

async function resetPassword(employeeNo) {
  const pwd = await askPassword(`重置 ${employeeNo} 的密码`);
  if (pwd === null) return;
  try {
    await api('/api/accounts/reset-password', {
      method: 'POST',
      body: JSON.stringify({ employee_no: employeeNo, new_password: pwd }),
    });
    alert('已重置');
  } catch (e) { alert('重置失败：' + e.message); }
}

async function toggleAccount(employeeNo, active) {
  if (!confirm(`${active ? '启用' : '停用'} ${employeeNo} 的登录账号？`)) return;
  try {
    await api('/api/accounts/active', {
      method: 'POST',
      body: JSON.stringify({ employee_no: employeeNo, active }),
    });
    await loadAccounts();
  } catch (e) { alert('操作失败：' + e.message); }
}

async function toggleEmployee(employeeNo, active) {
  if (!confirm(`${active ? '启用' : '停用'} ${employeeNo} 的员工档案？`)) return;
  try {
    await api('/api/employees/active', {
      method: 'POST',
      body: JSON.stringify({ employee_no: employeeNo, active }),
    });
    await loadAccounts();
  } catch (e) { alert('操作失败：' + e.message); }
}

async function changeOwnPassword() {
  const oldPwd = await askPassword('输入当前密码');
  if (oldPwd === null) return;
  const newPwd = await askPassword('设置新密码');
  if (newPwd === null) return;
  try {
    await api('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ old_password: oldPwd, new_password: newPwd }),
    });
    alert('密码已修改');
  } catch (e) { alert('修改失败：' + e.message); }
}

$('accRefreshBtn')?.addEventListener('click', loadAccounts);
$('changePwdBtn')?.addEventListener('click', changeOwnPassword);
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    if (tab.dataset.panel === 'account' && AUTH.is_manager) loadAccounts();
  });
});

/* ================================================================
 * 七、助教 Agent（只读）
 * ================================================================ */
function renderAgentTimeline(box, items, kind) {
  box.replaceChildren();
  if (!items || !items.length) {
    const tip = document.createElement('div');
    tip.className = 'empty-tip';
    tip.textContent = '（无）';
    box.appendChild(tip);
    return;
  }
  items.forEach((item, index) => {
    const div = document.createElement('div');
    div.className = 'agent-step';
    if (kind === 'plan') {
      div.textContent = `${index + 1}. ${item.tool}${item.reason ? ' · ' + item.reason : ''}`;
    } else {
      const denied = item.denied ? '（无权限）' : '';
      div.textContent = `${index + 1}. ${item.tool} · ${item.ok ? '成功' : '失败'}${denied} · ${Math.round(item.elapsed_ms || 0)}ms`;
      if (!item.ok) div.classList.add('bad');
    }
    box.appendChild(div);
  });
}

async function runAgent() {
  const question = $('agentQuestion').value.trim();
  if (!question) { $('agentHint').textContent = '请输入问题'; return; }

  const btn = $('agentRunBtn');
  btn.disabled = true;
  btn.textContent = '执行中…';
  $('agentResult').hidden = true;
  $('agentHint').textContent = 'Agent 正在执行…';

  try {
    const r = await api('/api/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
    $('agentHint').textContent = '';
    $('agentResult').hidden = false;
    renderAgentTimeline($('agentPlan'), r.plan || [], 'plan');
    renderAgentTimeline($('agentSteps'), r.steps || [], 'step');
    $('agentAnswer').innerHTML = mdToHtml(r.answer || '');
    $('agentMeta').textContent = r.refused
      ? ('已拒答：' + (r.refusal_reason || '未说明原因'))
      : ('引用：' + ((r.citations || []).join('、') || '无'));
  } catch (e) {
    $('agentHint').textContent = '执行失败：' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '执行';
  }
}

$('agentRunBtn')?.addEventListener('click', runAgent);
$('agentQuestion')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runAgent();
});
