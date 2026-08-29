// 获取问答表单，后续监听提交事件。
const askForm = document.querySelector("#ask-form");
// 获取问题输入框，用于读取用户问题和填入示例问题。
const questionInput = document.querySelector("#question");
// 获取问答提交按钮，用于请求期间禁用重复点击。
const askButton = document.querySelector("#ask-button");
// 获取问答结果面板，用于切换空状态和结果状态样式。
const resultPanel = document.querySelector("#result-panel");
// 获取尚未提问时显示的空状态。
const emptyState = document.querySelector("#empty-state");
// 获取问答请求执行期间显示的加载状态。
const loadingState = document.querySelector("#loading-state");
// 获取成功或失败后显示的答案区域。
const answerState = document.querySelector("#answer-state");
// 获取最终答案文字容器。
const answerElement = document.querySelector("#answer");
// 获取参考来源卡片列表容器。
const sourceList = document.querySelector("#source-list");
// 获取参考片段数量文字。
const sourceCount = document.querySelector("#source-count");
// 获取完整问答流程总耗时的显示元素。
const totalTimeElement = document.querySelector("#total-time");
// 获取 Query Rewrite 结果容器，用于展示实际送入 Qdrant 的查询。
const rewrittenQueryElement = document.querySelector("#rewritten-query");
// 获取当前处理阶段的用户可见说明。
const progressMessage = document.querySelector("#progress-message");
// 获取 Reasoning 开关状态文字。
const reasoningStatus = document.querySelector("#reasoning-status");
// 获取多轮消息列表。
const chatHistoryElement = document.querySelector("#chat-history");
// 获取新建会话按钮。
const newChatButton = document.querySelector("#new-chat");
// 获取知识分类选择框，用于限制 Qdrant 只检索指定分类。
const categorySelect = document.querySelector("#category");
// 获取顶部 Ollama/Qdrant 服务状态区域。
const serviceStatus = document.querySelector("#service-status");
// 获取“知识问答”标签按钮。
const qaTab = document.querySelector("#qa-tab");
// 获取“检索评估”标签按钮。
const evalTab = document.querySelector("#eval-tab");
// 获取完整问答页面区域。
const qaView = document.querySelector("#qa-view");
// 获取完整评估页面区域。
const evalView = document.querySelector("#eval-view");
// 获取运行评估按钮。
const runEvaluationButton = document.querySelector("#run-evaluation");
// 获取评估尚未执行时的空状态。
const evalEmpty = document.querySelector("#eval-empty");
// 获取评估执行期间的加载状态。
const evalLoading = document.querySelector("#eval-loading");
// 获取评估完成后的结果区域。
const evalResults = document.querySelector("#eval-results");
// 获取两种方法的汇总指标卡片容器。
const metricComparison = document.querySelector("#metric-comparison");
// 获取最终回答质量指标区域。
const generationMetrics = document.querySelector("#generation-metrics");
// 获取 BM25 加入前后差值汇总区域。
const bm25Impact = document.querySelector("#bm25-impact");
// 获取逐题评估表格的 tbody。
const caseList = document.querySelector("#case-list");

// 读取浏览器保存的会话 ID；第一次访问时创建新的 UUID。
let sessionId = localStorage.getItem("rag_session_id") || crypto.randomUUID();
// 保存当前会话 ID，使刷新页面后仍能恢复同一段对话。
localStorage.setItem("rag_session_id", sessionId);

// 定义 HTML 转义函数，防止来源文本被浏览器当成标签执行。
const escapeHtml = (value) => value.replace(/[&<>'"]/g, (char) => ({
  // 将 & 转换成 HTML 实体。
  "&": "&amp;",
  // 将小于号转换成 HTML 实体。
  "<": "&lt;",
  // 将大于号转换成 HTML 实体。
  ">": "&gt;",
  // 将单引号转换成 HTML 实体。
  "'": "&#39;",
  // 将双引号转换成 HTML 实体。
  '"': "&quot;",
// 使用当前字符从映射对象中取得替换值。
}[char]));

// 定义异步健康检查函数。
async function checkHealth() {
  // 捕获网络错误或非正常响应。
  try {
    // 请求后端 /health，同时检查 Ollama 和 Qdrant。
    const response = await fetch("/health");
    // 非 2xx 响应统一作为失败处理。
    if (!response.ok) throw new Error();
    // 设置在线样式，使状态点显示绿色。
    serviceStatus.className = "service-status is-online";
    // 更新用户可见的在线状态文字。
    serviceStatus.lastElementChild.textContent = "Ollama 与 Qdrant 已连接";
  // 捕获健康检查失败。
  } catch {
    // 设置错误样式，使状态点显示红色。
    serviceStatus.className = "service-status is-error";
    // 更新用户可见的失败状态文字。
    serviceStatus.lastElementChild.textContent = "本地服务连接失败";
  }
}

// 把一条用户或助手消息追加到聊天历史区域。
function appendChatMessage(role, text) {
  // 首次追加消息时移除“新会话”提示。
  chatHistoryElement.querySelector(".chat-empty")?.remove();
  // 创建安全的普通 div，不通过 innerHTML 插入消息正文。
  const message = document.createElement("div");
  // 根据角色设置左右不同的气泡样式。
  message.className = `chat-message is-${role}`;
  // 使用 textContent 防止消息中的 HTML 被执行。
  message.textContent = text;
  // 把消息添加到历史区域末尾。
  chatHistoryElement.appendChild(message);
  // 自动滚动到最新消息。
  chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
}

// 从后端内存恢复当前会话已有的历史消息。
async function loadChatHistory() {
  // 每次切换分类都先恢复空状态，防止残留其他分类的聊天内容。
  chatHistoryElement.innerHTML = '<p class="chat-empty">当前是新会话，提出第一个问题吧。</p>';
  // 请求 session_id 对应的历史。
  const response = await fetch(`/chat/${encodeURIComponent(sessionId)}?category=${encodeURIComponent(categorySelect.value)}`);
  // 请求失败时保留新会话空状态，不阻断页面其他功能。
  if (!response.ok) return;
  // 解析历史响应。
  const payload = await response.json();
  // 有历史时先清空默认提示。
  if (payload.turns.length) chatHistoryElement.innerHTML = "";
  // 按时间顺序恢复每轮用户和助手消息。
  payload.turns.forEach((turn) => {
    appendChatMessage("user", turn.question);
    appendChatMessage("assistant", turn.answer);
  });
}

// 从后端读取实际存在的知识分类并填充下拉框。
async function loadCategories() {
  // 请求由 knowledge 目录动态生成的分类列表。
  const response = await fetch("/categories");
  // 请求失败时保留 HTML 中的“全部”选项。
  if (!response.ok) return;
  // 解析分类数组。
  const payload = await response.json();
  // 读取用户上次选择，刷新后继续使用同一分类。
  const savedCategory = localStorage.getItem("rag_category") || "全部";
  // 清空硬编码占位项，由接口返回值完整重建。
  categorySelect.innerHTML = "";
  // 使用 DOM API 创建选项，避免分类名被当成 HTML。
  payload.categories.forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    categorySelect.appendChild(option);
  });
  // 只有保存的分类仍存在时才恢复，否则使用“全部”。
  categorySelect.value = payload.categories.includes(savedCategory) ? savedCategory : "全部";
}

// 根据 view 参数切换问答面板内部状态。
function setView(view) {
  // 只有 view 为 empty 时显示初始空状态。
  emptyState.hidden = view !== "empty";
  // 只有 view 为 loading 时显示加载动画。
  loadingState.hidden = view !== "loading";
  // 只有 view 为 answer 时显示答案和来源。
  answerState.hidden = view !== "answer";
  // 同步切换结果面板的空状态样式。
  resultPanel.classList.toggle("is-empty", view === "empty");
}

// 重置四个 RAG 阶段，避免新问题显示上一次请求的完成状态。
function resetPipeline() {
  // 遍历所有阶段并移除运行中和已完成样式。
  document.querySelectorAll("#pipeline li").forEach((item) => {
    // 清除阶段状态类。
    item.classList.remove("is-running", "is-completed");
    // 将非 Query Rewrite 阶段的详情恢复为等待执行。
    if (item.dataset.step !== "rewrite") item.querySelector("small").textContent = "等待执行";
  });
  // 恢复 Query Rewrite 的初始开关提示。
  reasoningStatus.textContent = "Reasoning: 等待后端配置";
  // 恢复初始进度说明。
  progressMessage.textContent = "准备执行 RAG 流程";
}

// 根据后端状态事件更新一个真实处理阶段。
function updatePipeline(event) {
  // 查找当前事件对应的阶段元素。
  const item = document.querySelector(`#pipeline li[data-step="${event.step}"]`);
  // 未知阶段不更新页面。
  if (!item) return;
  // running 时高亮当前阶段。
  item.classList.toggle("is-running", event.state === "running");
  // completed 时显示已完成样式。
  item.classList.toggle("is-completed", event.state === "completed");
  // Query Rewrite 完成时立即显示实际改写结果，其他阶段显示普通状态消息。
  const displayMessage = event.step === "rewrite" && event.state === "completed" && event.detail
    ? `改写为：${event.detail}`
    : event.message;
  // 更新阶段下方的状态详情。
  item.querySelector("small").textContent = displayMessage;
  // 更新进度区域主说明。
  progressMessage.textContent = displayMessage;
  // Query Rewrite 开始事件同时显示 reasoning 是否启用。
  if (event.step === "rewrite" && typeof event.reasoning_enabled === "boolean") {
    // 使用 ON 或 OFF 清晰展示后端真实配置。
    reasoningStatus.textContent = `Reasoning: ${event.reasoning_enabled ? "ON" : "OFF"} · ${event.message}`;
  }
}

// 把后端 sources 数组渲染成参考来源卡片。
function renderSources(sources) {
  // 显示进入 Prompt 的参考片段总数。
  sourceCount.textContent = `${sources.length} 个参考片段`;
  // 遍历来源并生成卡片 HTML，最后连接成一个字符串。
  sourceList.innerHTML = sources.map((source, index) => `
    <!-- 单个参考片段卡片。 -->
    <article class="source-card">
      <!-- 显示来源、片段编号和两个检索分数。 -->
      <div class="source-meta">
        <span class="source-name">[Reference ${index + 1}] ${escapeHtml(source.category)} · ${escapeHtml(source.point_name)} · ${escapeHtml(source.source)}</span>
        <span class="source-scores">${source.vector_score.toFixed(4)} / ${source.rerank_score.toFixed(4)}</span>
      </div>
      <!-- 显示经过 HTML 转义的知识片段正文。 -->
      <p class="source-text">${escapeHtml(source.text)}</p>
    </article>
  `).join("");
}

// 在知识问答和检索评估两个主页面之间切换。
function selectTab(tab) {
  // 判断当前目标是否为评估页。
  const evaluationSelected = tab === "evaluation";
  // 评估页未选中时，高亮知识问答标签。
  qaTab.classList.toggle("is-active", !evaluationSelected);
  // 评估页选中时，高亮检索评估标签。
  evalTab.classList.toggle("is-active", evaluationSelected);
  // 选中评估页时隐藏整个知识问答区域。
  qaView.hidden = evaluationSelected;
  // 未选中评估页时隐藏整个评估区域。
  evalView.hidden = !evaluationSelected;
}

// 把 0 到 1 的指标转换成一位小数百分比。
const percent = (value) => `${(value * 100).toFixed(1)}%`;
// 把毫秒数格式化成不带小数的可读文字。
const milliseconds = (value) => `${value.toFixed(0)} ms`;
// 带正负号显示指标差值，便于识别提升或下降。
const signed = (value, formatter) => `${value > 0 ? "+" : ""}${formatter(value)}`;

// 生成一种检索方法的六项指标卡片。
function methodCard(name, label, metrics, highlighted = false) {
  // 返回包含方法名、Hit@1、Hit@3、MRR 和耗时的 HTML。
  return `
    <article class="method-card${highlighted ? " is-reranker" : ""}">
      <div class="method-title"><strong>${name}</strong><span>${label}</span></div>
      <div class="metrics">
        <div class="metric"><small>HIT@1</small><b>${percent(metrics.hit_at_1)}</b></div>
        <div class="metric"><small>HIT@3</small><b>${percent(metrics.hit_at_3)}</b></div>
        <div class="metric"><small>HIT@5</small><b>${percent(metrics.hit_at_5)}</b></div>
        <div class="metric"><small>MRR</small><b>${metrics.mrr.toFixed(3)}</b></div>
        <div class="metric"><small>NDCG@5</small><b>${metrics.ndcg_at_5.toFixed(3)}</b></div>
        <div class="metric"><small>AVG TIME</small><b>${milliseconds(metrics.avg_latency_ms)}</b></div>
      </div>
    </article>`;
}

// 把数字排名转换成 #1 形式；未命中时显示横线。
function rankText(rank) {
  // 根据 rank 是否为 null 返回不同文字。
  return rank === null ? "—" : `#${rank}`;
}

// 根据重排前后排名生成变化样式和文字。
function changeLabel(qdrantRank, rerankerRank) {
  // 原来未命中、重排后命中时标记为新命中。
  if (qdrantRank === null && rerankerRank !== null) return ["improved", "新命中"];
  // 原来命中、重排后丢失时标记为退步。
  if (qdrantRank !== null && rerankerRank === null) return ["worse", "丢失"];
  // 两次排名完全相同时标记为持平。
  if (qdrantRank === rerankerRank) return ["same", "持平"];
  // 重排数字更小时表示排名提升，并显示提升名次。
  if (rerankerRank < qdrantRank) return ["improved", `↑ ${qdrantRank - rerankerRank}`];
  // 其余情况表示排名下降，并显示下降名次。
  return ["worse", `↓ ${rerankerRank - qdrantRank}`];
}

// 把 /evaluate 返回的完整数据渲染到评估页面。
function renderEvaluation(payload) {
  // 生成 Dense、Hybrid、Cross-Encoder 和 Query Rewrite 四张汇总指标卡。
  metricComparison.innerHTML = [
    // 第一张卡显示 Qdrant 原始向量排名指标。
    methodCard("Qdrant", "VECTOR RANKING", payload.summary.qdrant),
    // 第二张卡显示 Dense + BM25 的 RRF 融合排名。
    methodCard("Hybrid Search", "DENSE + BM25 + RRF", payload.summary.hybrid),
    // 第二张卡显示原问题经过 Cross-Encoder 重排后的指标。
    methodCard("Cross-Encoder", "ORIGINAL + RERANK", payload.summary.reranker),
    // 第三张卡显示 Query Rewrite 后重新召回并重排的完整管线。
    methodCard("Query Rewrite", "REWRITE + QDRANT + RERANK", payload.summary.rewrite_reranker, true),
  // 把三张卡片连接成 HTML 字符串。
  ].join("");
  // 明确展示加入 BM25 后相对于纯 Dense 的总体变化。
  const impact = payload.summary.bm25_impact;
  bm25Impact.innerHTML = `
    <article class="impact-card">
      <div class="impact-title"><small>BM25 BEFORE / AFTER</small><b>加入 BM25 带来的变化</b></div>
      <div><small>HIT@1</small><b>${signed(impact.hit_at_1_delta, percent)}</b></div>
      <div><small>HIT@3</small><b>${signed(impact.hit_at_3_delta, percent)}</b></div>
      <div><small>MRR</small><b>${signed(impact.mrr_delta, (value) => value.toFixed(3))}</b></div>
      <div><small>NDCG@5</small><b>${signed(impact.ndcg_at_5_delta, (value) => value.toFixed(3))}</b></div>
      <div><small>AVG TIME</small><b>${signed(impact.latency_delta_ms, milliseconds)}</b></div>
      <div><small>逐题变化</small><b>${impact.improved_cases} 升 / ${impact.same_cases} 平 / ${impact.worse_cases} 降</b></div>
    </article>`;
  // 单独显示生成答案命中率、引用率和完整 RAG 平均耗时。
  generationMetrics.innerHTML = `
    <article class="generation-card">
      <div><small>ANSWER MATCH</small><b>${percent(payload.summary.generation.answer_match_rate)}</b></div>
      <div><small>CITATION RATE</small><b>${percent(payload.summary.generation.citation_rate)}</b></div>
      <div><small>END-TO-END</small><b>${milliseconds(payload.summary.generation.avg_end_to_end_latency_ms)}</b></div>
    </article>`;
  // 遍历每个评估问题并生成表格行。
  caseList.innerHTML = payload.cases.map((item) => {
    // 比较原始 Qdrant 与 Query Rewrite 完整管线的最终排名变化。
    const [changeClass, changeText] = changeLabel(item.qdrant.rank, item.rewrite_reranker.rank);
    // 单独比较加入 BM25 前的 Dense 排名和加入后的 Hybrid 排名。
    const [bm25Class, bm25Text] = changeLabel(item.qdrant.rank, item.hybrid.rank);
    // 返回当前问题对应的表格行。
    return `
      <tr>
        <td><span class="case-question">${escapeHtml(item.question)}</span><span class="case-source">${escapeHtml(item.expected_source)}</span></td>
        <td><span class="rank-value">${rankText(item.qdrant.rank)}</span></td>
        <td><span class="rank-value">${rankText(item.hybrid.rank)}</span></td>
        <td><span class="change ${bm25Class}">${bm25Text}</span></td>
        <td><span class="rank-value">${rankText(item.reranker.rank)}</span></td>
        <td><span class="rank-value">${rankText(item.rewrite_reranker.rank)}</span></td>
        <td><span class="rewritten-eval-query">${escapeHtml(item.rewrite_reranker.rewritten_query)}</span></td>
        <td><span class="change ${changeClass}">${changeText}</span></td>
        <td><span class="latency">${milliseconds(item.rewrite_reranker.latency_ms)}</span></td>
      </tr>`;
  // 将所有表格行连接成 HTML 字符串。
  }).join("");
}

// 监听问答表单提交事件。
askForm.addEventListener("submit", async (event) => {
  // 阻止浏览器执行默认页面刷新提交。
  event.preventDefault();
  // 读取问题并去除首尾空格。
  const question = questionInput.value.trim();
  // 空问题不向后端发送请求。
  if (!question) return;
  // 请求期间禁用按钮，防止重复提交。
  askButton.disabled = true;
  // 立即把当前问题显示为用户消息。
  appendChatMessage("user", question);
  // 清除上一次请求留下的阶段状态。
  resetPipeline();
  // 显示问答加载状态。
  setView("loading");
  // 捕获后端错误或网络错误。
  try {
    // 向流式接口发送问题，后端会逐行返回真实阶段事件。
    const response = await fetch("/ask/stream", {
      // 使用 POST 方法提交问题。
      method: "POST",
      // 声明请求体是 JSON。
      headers: { "Content-Type": "application/json" },
      // 把问题和当前会话 ID 转换成 JSON 字符串。
      body: JSON.stringify({ question, session_id: sessionId, category: categorySelect.value }),
    });
    // 非 2xx 响应直接读取错误文字并进入 catch。
    if (!response.ok) throw new Error(await response.text() || "请求失败");
    // 取得响应体读取器，用于边接收边更新页面。
    const reader = response.body.getReader();
    // 创建 UTF-8 解码器处理网络字节。
    const decoder = new TextDecoder();
    // 保存尚未形成完整一行的部分内容。
    let buffer = "";
    // 保存最后收到的 result 事件。
    let payload = null;
    // 持续读取直到后端关闭响应流。
    while (true) {
      // 读取下一批网络数据。
      const { value, done } = await reader.read();
      // 将当前字节追加到文本缓冲区。
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      // 按换行拆分完整事件，并保留最后一个不完整片段。
      const lines = buffer.split("\n");
      // 最后一项可能尚未传输完整，放回缓冲区。
      buffer = lines.pop();
      // 处理当前已经完整到达的所有事件。
      lines.filter(Boolean).forEach((line) => {
        // 把一行 JSON 转换成事件对象。
        const eventPayload = JSON.parse(line);
        // 状态事件用于更新处理步骤。
        if (eventPayload.type === "status") updatePipeline(eventPayload);
        // 最终事件保存为答案载荷。
        if (eventPayload.type === "result") payload = eventPayload;
      });
      // 流读取完成时跳出循环。
      if (done) break;
    }
    // 没有收到最终结果说明流在处理中意外中断。
    if (!payload) throw new Error("响应流结束，但没有收到最终答案");
    // 恢复普通答案样式，清除之前可能存在的错误颜色。
    answerElement.className = "answer";
    // 使用 textContent 安全显示模型答案。
    answerElement.textContent = payload.answer;
    // 将最终回答追加到多轮聊天历史。
    appendChatMessage("assistant", payload.answer);
    // 显示经过 Query Rewrite 后实际用于向量召回的查询。
    rewrittenQueryElement.textContent = payload.rewritten_query;
    // 把后端统计的毫秒转换成秒，并保留两位小数。
    totalTimeElement.textContent = `总耗时 ${(payload.elapsed_ms / 1000).toFixed(2)} 秒`;
    // 渲染后端返回的参考来源。
    renderSources(payload.sources);
  // 捕获问答请求错误。
  } catch (error) {
    // 给答案区域添加错误样式。
    answerElement.className = "answer error-message";
    // 显示可读的失败原因。
    answerElement.textContent = `查询失败：${error.message}`;
    // 请求失败时清空上一次的 Query Rewrite 结果。
    rewrittenQueryElement.textContent = "";
    // 请求失败时不显示上一次请求的耗时。
    totalTimeElement.textContent = "";
    // 清空上一次可能存在的来源卡片。
    renderSources([]);
  // 无论成功失败都执行收尾处理。
  } finally {
    // 恢复问答按钮可点击状态。
    askButton.disabled = false;
    // 切换到答案区域，让成功或错误信息可见。
    setView("answer");
  }
});

// 点击“新建会话”时清除后端旧历史并生成新的 session_id。
newChatButton.addEventListener("click", async () => {
  // 请求后端删除当前会话的内存历史。
  await fetch(`/chat/${encodeURIComponent(sessionId)}?category=${encodeURIComponent(categorySelect.value)}`, { method: "DELETE" });
  // 创建全新的浏览器会话标识。
  sessionId = crypto.randomUUID();
  // 保存新标识供刷新页面后继续使用。
  localStorage.setItem("rag_session_id", sessionId);
  // 清空页面历史并显示新会话提示。
  chatHistoryElement.innerHTML = '<p class="chat-empty">当前是新会话，提出第一个问题吧。</p>';
  // 清空问题输入框和上一次答案区域。
  questionInput.value = "";
  // 恢复问答结果初始状态。
  setView("empty");
  // 把输入焦点交还给用户。
  questionInput.focus();
});

// 切换分类时读取该分类独立的聊天记录，并清空上一分类的答案面板。
categorySelect.addEventListener("change", async () => {
  // 保存分类选择供刷新页面后恢复。
  localStorage.setItem("rag_category", categorySelect.value);
  // 加载 session_id 与当前分类组合后的独立历史。
  await loadChatHistory();
  // 清空问题并恢复等待状态，避免误把旧答案理解为当前分类结果。
  questionInput.value = "";
  setView("empty");
});

// 查找所有带 data-question 的示例问题按钮。
document.querySelectorAll("[data-question]").forEach((button) => {
  // 为当前示例按钮注册点击事件。
  button.addEventListener("click", () => {
    // 将按钮保存的问题填入输入框。
    questionInput.value = button.dataset.question;
    // 将键盘焦点移动到输入框，方便继续编辑。
    questionInput.focus();
  });
});

// 点击知识问答标签时显示问答页并隐藏评估页。
qaTab.addEventListener("click", () => selectTab("qa"));
// 点击检索评估标签时显示评估页并隐藏问答页。
evalTab.addEventListener("click", () => selectTab("evaluation"));
// 监听运行评估按钮点击事件。
runEvaluationButton.addEventListener("click", async () => {
  // 评估期间禁用按钮，避免重复并行运行模型。
  runEvaluationButton.disabled = true;
  // 隐藏尚未运行提示。
  evalEmpty.hidden = true;
  // 隐藏上一次评估结果。
  evalResults.hidden = true;
  // 显示评估加载动画。
  evalLoading.hidden = false;
  // 捕获评估接口或网络错误。
  try {
    // 调用 FastAPI /evaluate 执行全部评估问题。
    const response = await fetch("/evaluate", { method: "POST" });
    // 解析后端返回的评估 JSON。
    const payload = await response.json();
    // 非 2xx 响应转换成异常。
    if (!response.ok) throw new Error(payload.detail || "评估失败");
    // 将汇总指标和逐题结果渲染到页面。
    renderEvaluation(payload);
    // 显示评估结果区域。
    evalResults.hidden = false;
  // 捕获评估失败。
  } catch (error) {
    // 在空状态区域安全显示错误原因。
    evalEmpty.innerHTML = `<div><h2 class="error-message">评估失败</h2><p>${escapeHtml(error.message)}</p></div>`;
    // 显示包含错误信息的区域。
    evalEmpty.hidden = false;
  // 无论成功失败都执行收尾处理。
  } finally {
    // 隐藏加载动画。
    evalLoading.hidden = true;
    // 恢复评估按钮可点击状态。
    runEvaluationButton.disabled = false;
  }
});

// 页面加载后立即检查 Ollama 和 Qdrant 是否可用。
checkHealth();
// 页面加载后先取得分类，再恢复当前分类与 session_id 对应的历史消息。
loadCategories().then(loadChatHistory);
