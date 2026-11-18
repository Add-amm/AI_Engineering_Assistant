import { loadChats, removeChat, saveChat } from "./chat-storage.js";

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const labels = { assistant: "Assistant", retrieval: "Retrieval lab", library: "Source library" };
const topics = {
  python: { label: "Python", icon: "code" },
  fastapi: { label: "FastAPI", icon: "zap" },
  transformers: { label: "Transformers", icon: "sparkles" },
  rag: { label: "RAG & retrieval", icon: "network" },
};
const methodNames = { bm25: "BM25", dense: "Dense vectors", hybrid: "Hybrid (RRF)", reranked: "Hybrid + rerank" };
const generationNames = { extractive: "Local excerpts", ollama: "Ollama", openai: "Hosted model" };
const stored = loadChats();
const state = { view: "", topic: "", sources: null, health: null, libraryError: false, searchRequest: null };
const chats = stored.chats;
const pending = new Map();
const drafts = new Map();
const unsaved = new Set();
let activeChatId = "";
let managedChatId = "";

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (typeof content === "string") node.textContent = content;
  return node;
}

function icon(name) {
  const node = document.createElementNS(SVG, "svg");
  node.setAttribute("class", "icon");
  node.setAttribute("aria-hidden", "true");
  const glyph = document.createElementNS(SVG, "use");
  glyph.setAttribute("href", "#i-" + name);
  node.append(glyph);
  return node;
}

function topicIcon(topic, fallback = "book") {
  const badge = element("span", "topic-icon " + (topics[topic] ? topic : ""));
  badge.append(icon(topics[topic]?.icon || fallback));
  return badge;
}

function externalLink(url, className, content) {
  const link = element("a", className, content);
  try {
    const safe = new URL(url);
    if (safe.protocol === "https:") link.href = safe.href;
  } catch {
    // The link label still renders if source metadata is unavailable.
  }
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function appendInline(parent, text, citations = new Map()) {
  const tokens = /`([^`\n]+)`|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|\[(\d+)\]/g;
  let offset = 0;
  for (const match of text.matchAll(tokens)) {
    parent.append(text.slice(offset, match.index));
    const source = citations.get(Number(match[4]));
    if (match[1]) parent.append(element("code", "", match[1]));
    else if (match[2]) {
      const strong = element("strong");
      appendInline(strong, match[2], citations);
      parent.append(strong);
    } else if (match[3]) parent.append(element("em", "", match[3]));
    else if (source) {
      const link = externalLink(source.url, "cite-chip", match[4]);
      link.setAttribute("aria-label", "Citation " + match[4] + " · " + source.title);
      link.title = source.title;
      parent.append(link);
    } else parent.append(match[0]);
    offset = match.index + match[0].length;
  }
  parent.append(text.slice(offset));
}

function markdown(text, citations = new Map()) {
  const result = document.createDocumentFragment();
  const lines = text.split("\n");
  const fence = /^\s*```[\w+-]*\s*$/;
  const unordered = /^\s*[-*+]\s+(.+)/;
  const ordered = /^\s*\d+\.\s+(.+)/;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    if (fence.test(line)) {
      const code = [];
      while (++i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i]);
      const block = element("pre");
      block.tabIndex = 0;
      block.append(element("code", "", code.join("\n")));
      result.append(block);
      continue;
    }
    const heading = /^(#{1,6})\s+(.*)/.exec(line);
    if (heading) {
      const title = element(heading[1].length > 2 ? "h4" : "h3");
      appendInline(title, heading[2], citations);
      result.append(title);
      continue;
    }
    const listPattern = unordered.test(line) ? unordered : ordered.test(line) ? ordered : null;
    if (listPattern) {
      const list = element(listPattern === unordered ? "ul" : "ol");
      while (i < lines.length) {
        const entry = listPattern.exec(lines[i]);
        if (!entry) break;
        const item = element("li");
        appendInline(item, entry[1], citations);
        list.append(item);
        i++;
      }
      i--;
      result.append(list);
      continue;
    }
    const paragraph = element("p");
    appendInline(paragraph, line, citations);
    result.append(paragraph);
  }
  return result;
}

function message(title, detail, kind = "info") {
  const box = element("div", "message-box " + kind);
  const copy = element("div");
  copy.append(element("strong", "", title), element("p", "", detail));
  box.append(kind === "loading" ? element("span", "spinner") : icon(kind === "error" ? "alert" : "search"), copy);
  return box;
}

async function jsonRequest(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error("Request failed (" + response.status + ")");
  return response.json();
}

const smallScreen = window.matchMedia("(max-width: 820px)");

function setMenu(open, restoreFocus = true) {
  const expanded = smallScreen.matches && open;
  $("sidebar").classList.toggle("is-open", expanded);
  $("sidebar").inert = smallScreen.matches && !expanded;
  document.querySelector(".workspace").inert = expanded;
  $("sidebar-scrim").hidden = !expanded;
  $("menu-button").setAttribute("aria-expanded", String(expanded));
  if (expanded) $("sidebar-close").focus();
  else if (restoreFocus && smallScreen.matches) $("menu-button").focus();
}

function chatRoute(id = "") {
  return id ? "#assistant/" + id : "#assistant";
}

function navigate() {
  const [name, filter] = location.hash.slice(1).split("/");
  if (name === "main-content" && state.view) return;
  const view = Object.hasOwn(labels, name) ? name : "assistant";
  const changed = state.view !== view;
  state.view = view;
  state.topic = view === "library" && Object.hasOwn(topics, filter) ? filter : "";
  document.querySelector(".workspace").classList.toggle("is-chat", view === "assistant");
  for (const key of Object.keys(labels)) $("view-" + key).hidden = key !== view;
  document.querySelectorAll("[data-view]").forEach((link) => {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });

  const hadOpenMenu = $("sidebar").classList.contains("is-open");
  setMenu(false, false);
  if (view === "assistant") {
    const selected = chats.has(filter) ? filter : "";
    if (selected !== activeChatId || changed) {
      activeChatId = selected;
      restoreDraft();
    }
    renderChat(changed || selected !== $("chat-thread").dataset.chat);
  } else {
    $("page-label").textContent = labels[view];
    $("page-label").removeAttribute("title");
    document.title = "DocLens · " + labels[view];
    if (view === "library") renderLibrary();
  }
  renderHistory();
  if (changed) window.scrollTo(0, 0);
  if (hadOpenMenu) $("main-content").focus();
}

function newChat() {
  drafts.delete("");
  $("history-query").value = "";
  location.hash = chatRoute();
  navigate();
  restoreDraft();
  $("ask-question").focus();
}

function passageList(evidence) {
  const list = element("div", "evidence-list");
  for (const hit of evidence) {
    const details = element("details", "evidence-card");
    const summary = element("summary");
    const title = element("span");
    title.append(element("strong", "", hit.title));
    if (hit.heading.length) title.append(element("small", "", hit.heading.join(" › ")));
    const score = element("span", "passage-score", Number(hit.score).toFixed(3));
    score.title = "Raw score for the selected retrieval method";
    summary.append(element("span", "rank-number", String(hit.number)), title, score, icon("chevron-right"));
    const body = element("div", "passage-body");
    const text = element("div", "passage-text");
    text.append(markdown(hit.text));
    const link = externalLink(hit.url, "source-anchor", "View source · lines " + hit.start + "–" + hit.end);
    link.append(icon("arrow-up-right"));
    body.append(text, link);
    details.append(summary, body);
    list.append(details);
  }
  return list;
}

function copyButton(answer) {
  const copy = element("button", "icon-button");
  copy.type = "button";
  copy.setAttribute("aria-label", "Copy answer");
  copy.title = "Copy answer";
  copy.append(icon("copy"));
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answer);
      copy.title = "Answer copied";
      copy.setAttribute("aria-label", "Answer copied");
      copy.replaceChildren(icon("check"));
    } catch {
      copy.title = "Clipboard unavailable. Select the answer text to copy it.";
      copy.setAttribute("aria-label", copy.title);
    }
  });
  return copy;
}

function citationPanel(citations) {
  const panel = element("aside", "citation-panel");
  const links = element("div", "citation-links");
  panel.append(element("h3", "", "Sources cited"), links);
  for (const citation of citations) {
    const link = externalLink(citation.url, "citation-link");
    link.title = citation.heading.at(-1) || citation.version;
    link.append(
      element("span", "citation-number", String(citation.number)),
      element("span", "citation-copy", citation.title),
      icon("arrow-up-right"),
    );
    links.append(link);
  }
  return panel;
}

function answerCard(turn) {
  const data = turn.result;
  const card = element("article", "answer-card");
  const heading = element("div", "answer-head");
  const answerLabel = element("div", "answer-label");
  const avatar = element("span", "brand-icon");
  avatar.append(icon("logo"));
  const name = element("div");
  name.append(
    element("h2", "", data.abstained ? "Insufficient evidence" : "DocLens"),
    element("p", "answer-mode", generationNames[data.generation] || data.generation),
  );
  answerLabel.append(avatar, name);
  heading.append(answerLabel, copyButton(data.answer));
  const content = element("div", "answer-body");
  content.append(markdown(data.answer, new Map(data.citations.map((cite) => [cite.number, cite]))));
  card.append(heading, content);
  if (data.note) card.append(element("p", "notice", data.note));
  if (data.citations.length) card.append(citationPanel(data.citations));
  if (data.evidence.length) {
    const detail = element("details", "answer-evidence");
    const openLab = element("button", "link-button", "Open in retrieval lab");
    openLab.type = "button";
    openLab.append(icon("arrow-up-right"));
    openLab.addEventListener("click", () => {
      $("search-question").setCustomValidity("");
      $("search-question").value = (data.retrieval_query || turn.question).replace(/\s+/g, " ");
      $("search-topic").value = turn.topic || "";
      $("search-mode").value = data.mode;
      location.hash = "retrieval";
      navigate();
      $("search-form").requestSubmit();
    });
    detail.append(element("summary", "", "Inspect retrieved passages · " + data.evidence.length), openLab);
    detail.addEventListener("toggle", () => {
      if (detail.open && !detail.querySelector(".evidence-list")) detail.append(passageList(data.evidence));
    });
    card.append(detail);
  }
  return card;
}

function updateStorageStatus() {
  const status = $("history-error");
  status.hidden = stored.available && !unsaved.size;
  status.textContent = status.hidden ? "" : "Some chats could not be saved in this browser.";
  $("chat-footer").textContent = status.hidden ? "Chats are saved in this browser." : "History storage is unavailable for some chats.";
}

function persist(chat) {
  try {
    saveChat(chat);
    stored.available = true;
    unsaved.delete(chat.id);
  } catch {
    unsaved.add(chat.id);
  }
  updateStorageStatus();
}

function renderHistory() {
  const filter = $("history-query").value.trim().toLowerCase();
  $("history-search").hidden = chats.size === 0;
  const list = $("history-list");
  const previousScroll = list.scrollTop;
  const rows = [];
  const sorted = [...chats.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const chat of sorted) {
    if (filter && !chat.title.toLowerCase().includes(filter) &&
      !chat.turns.some((turn) => turn.question.toLowerCase().includes(filter) ||
        turn.result?.answer.toLowerCase().includes(filter))) continue;
    const isCurrent = state.view === "assistant" && activeChatId === chat.id;
    const row = element("li", isCurrent ? "history-row is-active" : "history-row");
    const link = element("a", "history-link");
    link.href = chatRoute(chat.id);
    if (isCurrent) link.setAttribute("aria-current", "page");
    link.title = chat.title;
    link.append(icon("chat"), element("span", "", chat.title));
    const more = element("button", "icon-button history-menu");
    more.type = "button";
    more.dataset.manageId = chat.id;
    more.setAttribute("aria-label", "Manage chat: " + chat.title);
    more.title = "Chat options";
    more.append(icon("more"));
    more.addEventListener("click", () => manageChat(chat));
    row.append(link, more);
    rows.push(row);
  }
  $("history-empty").hidden = rows.length > 0;
  $("history-empty").textContent = filter ? "No matching conversations." : "Your conversations will appear here.";
  list.replaceChildren(...rows);
  list.scrollTop = previousScroll;
  document.querySelector('[data-view="assistant"]').href = chatRoute(activeChatId);
  document.querySelector(".brand").href = chatRoute(activeChatId);
}

function reserveId() {
  return globalThis.crypto.randomUUID?.() ?? Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
}

function adjustComposer() {
  const input = $("ask-question");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  $("ask-submit").disabled = input.value.trim().length < 2;
}

function preserveDraft() {
  const input = $("ask-question");
  input.setCustomValidity("");
  drafts.set(activeChatId, { text: input.value, topic: $("ask-topic").value });
  adjustComposer();
}

function restoreDraft() {
  const draft = drafts.get(activeChatId);
  const input = $("ask-question");
  input.setCustomValidity("");
  input.value = draft?.text || "";
  $("ask-topic").value = draft?.topic ?? chats.get(activeChatId)?.turns.at(-1)?.topic ?? "";
  adjustComposer();
}

function makeTurn(chat, turn, kind, retryable) {
  const row = element("li", "chat-turn");
  row.dataset.id = turn.id;
  row.dataset.kind = kind;
  row.dataset.retryable = String(retryable);
  const question = element("div", "user-message");
  question.append(element("span", "visually-hidden", "You: "), element("span", "", turn.question));
  row.append(question);
  if (kind === "ready") {
    row.append(answerCard(turn));
  } else if (kind === "pending") {
    row.append(message("Searching the documentation", "Finding passages in the indexed sources…", "loading"));
  } else {
    const problem = turn.error === "stopped" ? "Response stopped" : "Response unavailable";
    const details = turn.error === "failed" ? "The API couldn’t finish this request." : "Your question is still in this chat.";
    const failure = message(problem, details, "error");
    if (retryable) {
      const retry = element("button", "link-button", "Try again");
      retry.type = "button";
      retry.addEventListener("click", () => requestAnswer(chat, turn));
      failure.lastElementChild.append(retry);
    }
    row.append(failure);
  }
  return row;
}

function renderChat(scrollToEnd = false) {
  if (state.view !== "assistant") return;
  const chat = chats.get(activeChatId);
  const title = chat?.title || "New chat";
  $("page-label").textContent = title;
  $("page-label").title = title;
  document.title = "DocLens · " + title;
  const scroll = $("chat-scroll");
  const nearEnd = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 130;
  const thread = $("chat-thread");
  if (thread.dataset.chat !== (chat?.id || "")) {
    thread.dataset.chat = chat?.id || "";
    thread.replaceChildren();
  }
  $("view-assistant").classList.toggle("has-thread", Boolean(chat));
  $("chat-welcome").hidden = Boolean(chat);
  $("suggestions").hidden = Boolean(chat);
  thread.hidden = !chat;
  scroll.tabIndex = chat ? 0 : -1;
  $("ask-question").placeholder = chat ? "Ask a follow-up…" : "Ask a question about the docs…";

  const running = pending.get(activeChatId);
  $("ask-submit").hidden = Boolean(running);
  $("ask-stop").hidden = !running;
  if (!chat) return;

  const entries = new Map(Array.from(thread.children, (node) => [node.dataset.id, node]));
  for (const turn of chat.turns) {
    const retryable = !turn.result && !running && turn === chat.turns.at(-1);
    const kind = turn.result ? "ready" : running?.turnId === turn.id ? "pending" : turn.error || "interrupted";
    const existing = entries.get(turn.id);
    if (existing?.dataset.kind === kind && existing.dataset.retryable === String(retryable)) continue;
    const row = makeTurn(chat, turn, kind, retryable);
    if (existing) existing.replaceWith(row);
    else thread.append(row);
  }
  if (scrollToEnd || nearEnd) scroll.scrollTop = scroll.scrollHeight;
}

function questionValue(input) {
  const question = input.value.trim();
  if (question.length >= 2) return question;
  input.setCustomValidity("Enter at least two characters.");
  input.reportValidity();
  return "";
}

function submitChat(event) {
  event.preventDefault();
  if (pending.has(activeChatId)) return;
  const question = questionValue($("ask-question"));
  if (!question) return;
  let chat = chats.get(activeChatId);
  if (!chat) {
    const summary = Array.from(question.replace(/\s+/g, " "));
    const title = summary.length > 68 ? summary.slice(0, 67).join("").trimEnd() + "…" : summary.join("");
    chat = { id: reserveId(), title, updatedAt: Date.now(), turns: [] };
    chats.set(chat.id, chat);
    drafts.delete("");
    activeChatId = chat.id;
    history.replaceState(null, "", chatRoute(chat.id));
  }
  const turn = { id: reserveId(), question, topic: $("ask-topic").value, result: null, error: null };
  chat.turns.push(turn);
  chat.updatedAt = Date.now();
  drafts.delete(chat.id);
  persist(chat);
  restoreDraft();
  renderHistory();
  requestAnswer(chat, turn);
  $("ask-question").focus({ preventScroll: true });
}

async function requestAnswer(chat, turn) {
  if (pending.has(chat.id)) return;
  const request = { controller: new AbortController(), turnId: turn.id };
  pending.set(chat.id, request);
  turn.error = null;
  if (activeChatId === chat.id) renderChat(true);
  const previous = chat.turns.slice(0, chat.turns.indexOf(turn))
    .filter((entry) => entry.result && !entry.result.abstained)
    .slice(-6)
    .map((entry) => ({ question: entry.question, answer: Array.from(entry.result.answer).slice(0, 2500).join("") }));
  try {
    const result = await postQuery("/ask", {
      question: turn.question, topic: turn.topic || null, history: previous,
    }, request.controller.signal);
    if (pending.get(chat.id) !== request) return;
    turn.result = result;
  } catch {
    if (pending.get(chat.id) !== request) return;
    turn.error = "failed";
  } finally {
    if (pending.get(chat.id) === request) {
      pending.delete(chat.id);
      persist(chat);
      if ($("history-query").value) renderHistory();
      if (activeChatId === chat.id) renderChat();
    }
  }
}

function stopResponse() {
  const running = pending.get(activeChatId);
  if (!running) return;
  pending.delete(activeChatId);
  running.controller.abort();
  const chat = chats.get(activeChatId);
  chat.turns.find((turn) => turn.id === running.turnId).error = "stopped";
  persist(chat);
  renderChat();
  $("ask-question").focus({ preventScroll: true });
}

function manageChat(chat) {
  managedChatId = chat.id;
  $("chat-name").value = chat.title;
  $("chat-name").setCustomValidity("");
  $("manage-title").textContent = "Rename chat";
  $("rename-chat").hidden = false;
  $("delete-chat").hidden = true;
  $("manage-chat").showModal();
  $("chat-name").select();
}

function finishManage(focusChat = true) {
  const id = managedChatId;
  $("manage-chat").close();
  managedChatId = "";
  if (focusChat) document.querySelector('[data-manage-id="' + id + '"]')?.focus();
}

function renameChat(event) {
  event.preventDefault();
  const chat = chats.get(managedChatId);
  if (!chat) return;
  const title = $("chat-name").value.trim();
  if (!title) {
    $("chat-name").setCustomValidity("Enter a chat name.");
    $("chat-name").reportValidity();
    return;
  }
  chat.title = title;
  persist(chat);
  renderHistory();
  renderChat();
  finishManage();
}

function showDeleteChat() {
  const chat = chats.get(managedChatId);
  if (!chat) return;
  $("manage-title").textContent = "Delete chat?";
  $("delete-chat-description").textContent = "The messages in “" + chat.title + "” will be removed from this browser.";
  $("rename-chat").hidden = true;
  $("delete-chat").hidden = false;
  $("cancel-delete-chat").focus();
}

function deleteChat(event) {
  event.preventDefault();
  const id = managedChatId;
  try {
    removeChat(id);
    unsaved.delete(id);
  } catch {
    $("history-error").hidden = false;
    $("history-error").textContent = "This chat could not be deleted from browser storage.";
    finishManage();
    return;
  }
  pending.get(id)?.controller.abort();
  pending.delete(id);
  drafts.delete(id);
  chats.delete(id);
  finishManage(false);
  updateStorageStatus();
  if (activeChatId === id) {
    activeChatId = "";
    if (state.view === "assistant") {
      history.replaceState(null, "", chatRoute());
      restoreDraft();
      renderChat();
    }
  }
  if (!chats.size) $("history-query").value = "";
  renderHistory();
  $("new-chat").focus();
}

function searchResult(root, data) {
  if (!data.evidence.length) {
    root.replaceChildren(message("No passages found", "Try another term or search across all topics."));
    return;
  }
  const heading = element("div", "section-header");
  const count = data.evidence.length + (data.evidence.length === 1 ? " passage" : " passages");
  heading.append(element("h2", "", "Retrieved passages"));
  heading.append(element("p", "result-meta", count + " · " + methodNames[data.mode] + " · " + data.elapsed_ms + " ms"));
  root.replaceChildren(heading, passageList(data.evidence));
}

function postQuery(path, body, signal) {
  return jsonRequest(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

async function submitSearch(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const question = questionValue($("search-question"));
  if (!question) return;
  const values = new FormData(form);
  const body = { question, topic: values.get("topic") || null, mode: values.get("mode"), top_k: Number(values.get("top_k")) };
  state.searchRequest?.abort();
  const controller = new AbortController();
  state.searchRequest = controller;

  const root = $("search-output");
  const button = $("search-submit");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  root.hidden = false;
  root.replaceChildren(message("Searching the documentation", "Finding passages in the indexed sources…", "loading"));
  try {
    const result = await postQuery("/search", body, controller.signal);
    if (state.searchRequest === controller) searchResult(root, result);
  } catch {
    if (state.searchRequest !== controller) return;
    const failure = message("Couldn’t complete the request", "Check that the API is running, then try again.", "error");
    const retry = element("button", "link-button", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => form.requestSubmit());
    failure.lastElementChild.append(retry);
    root.replaceChildren(failure);
  } finally {
    if (state.searchRequest === controller) {
      state.searchRequest = null;
      button.disabled = false;
      button.removeAttribute("aria-busy");
    }
  }
}

function collectionCard(group, documents) {
  const card = element("article", "collection-card");
  const top = element("div", "collection-heading");
  const copy = element("div");
  copy.append(
    element("h2", "", group.title),
    element("p", "", group.description),
    element("div", "collection-version", group.version + " · " + group.license + " · " + group.revision.slice(0, 8)),
  );
  const repo = externalLink(group.url, "source-anchor", "Snapshot");
  repo.append(icon("arrow-up-right"));
  top.append(topicIcon(group.topic), copy, repo);
  const pages = element("div", "collection-pages");
  for (const doc of documents) {
    const link = externalLink(doc.url, "document-link");
    const arrow = icon("arrow-up-right");
    arrow.classList.add("outbound");
    link.append(icon("file"), element("span", "", doc.title), arrow);
    pages.append(link);
  }
  card.append(top, pages);
  return card;
}

function renderLibrary() {
  document.querySelectorAll("[data-library-topic]").forEach((link) => {
    if (link.dataset.libraryTopic === state.topic) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  });
  const root = $("library-output");
  if (!state.sources) {
    if (!state.libraryError) {
      root.replaceChildren(message("Loading the source library", "Reading the pinned source inventory…", "loading"));
      return;
    }
    const failure = message("Source library unavailable", "The API couldn’t load the source inventory.", "error");
    const retry = element("button", "link-button", "Reload sources");
    retry.type = "button";
    retry.addEventListener("click", loadLibrary);
    failure.lastElementChild.append(retry);
    root.replaceChildren(failure);
    return;
  }
  const query = $("library-query").value.trim().toLowerCase();
  const cards = [];
  let count = 0;
  for (const group of state.sources.collections) {
    if (state.topic && state.topic !== group.topic) continue;
    const documents = state.sources.documents.filter((doc) => doc.collection === group.id &&
      [doc.title, group.title, doc.version, doc.license].some((value) => value.toLowerCase().includes(query)));
    if (!documents.length) continue;
    cards.push(collectionCard(group, documents));
    count += documents.length;
  }
  if (!count) {
    root.replaceChildren(message("No matching documents", "Try a different title or choose another topic."));
    return;
  }
  const summary = count + (count === 1 ? " document" : " documents") + " · Snapshot retrieved " + state.sources.retrieved_on;
  root.replaceChildren(element("p", "library-count", summary), ...cards);
}

async function loadLibrary() {
  state.libraryError = false;
  if (state.view === "library") renderLibrary();
  try {
    state.sources = await jsonRequest("/sources");
  } catch {
    state.libraryError = true;
  }
  if (state.view === "library") renderLibrary();
}

async function loadHealth() {
  try {
    state.health = await jsonRequest("/health");
    $("index-status").dataset.state = "ready";
    $("index-label").textContent = "Index ready";
    $("index-status").title = state.health.documents + " indexed documents · " + state.health.passages + " passages";
    $("footer-index").textContent = state.health.documents + " documents · " + state.health.passages + " indexed passages";
    $("search-index-info").replaceChildren(
      element("span", "", state.health.passages + " indexed passages"),
      element("span", "", "Vectors · " + state.health.embeddings),
      element("span", "", "Reranker · " + state.health.reranker),
    );
    $("search-index-info").hidden = false;
    $("dialog-mode").textContent = state.health.generation === "extractive"
      ? "The current setup answers with relevant excerpts and runs without a model or API key."
      : "Current answer backend · " + generationNames[state.health.generation] + " · " + state.health.model;
  } catch {
    $("index-status").dataset.state = "error";
    $("index-label").textContent = "Index offline";
  }
}

window.addEventListener("hashchange", navigate);
smallScreen.addEventListener("change", () => setMenu(false, false));
$("menu-button").addEventListener("click", () => setMenu(true));
$("sidebar-close").addEventListener("click", () => setMenu(false));
$("sidebar-scrim").addEventListener("click", () => setMenu(false));
$("sidebar").addEventListener("click", (event) => {
  if (event.target.closest("a")?.hash === location.hash && $("sidebar").classList.contains("is-open")) setMenu(false);
});
$("new-chat").addEventListener("click", newChat);
$("help-open").addEventListener("click", () => {
  if (smallScreen.matches) setMenu(false);
  $("help-dialog").showModal();
});
$("ask-form").addEventListener("submit", submitChat);
$("ask-stop").addEventListener("click", stopResponse);
$("ask-question").addEventListener("input", preserveDraft);
$("ask-topic").addEventListener("change", preserveDraft);
$("search-form").addEventListener("submit", submitSearch);
$("search-question").addEventListener("input", (event) => event.target.setCustomValidity(""));
$("library-query").addEventListener("input", renderLibrary);
$("history-query").addEventListener("input", renderHistory);
$("rename-chat").addEventListener("submit", renameChat);
$("chat-name").addEventListener("input", (event) => event.target.setCustomValidity(""));
$("show-delete-chat").addEventListener("click", showDeleteChat);
$("cancel-delete-chat").addEventListener("click", () => {
  $("manage-title").textContent = "Rename chat";
  $("delete-chat").hidden = true;
  $("rename-chat").hidden = false;
  $("show-delete-chat").focus();
});
$("delete-chat").addEventListener("submit", deleteChat);
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    $("ask-question").value = button.dataset.question;
    $("ask-topic").value = "";
    preserveDraft();
    $("ask-form").requestSubmit();
  });
});
$("ask-question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!$("ask-submit").disabled && !pending.has(activeChatId)) $("ask-form").requestSubmit();
  }
});
document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]")) return;
  if (event.key === "Escape" && $("sidebar").classList.contains("is-open")) setMenu(false);
  if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    if ($("sidebar").classList.contains("is-open")) setMenu(false, false);
    if (state.view === "assistant") $("ask-question").focus();
    else if (state.view === "retrieval") $("search-question").focus();
    else $("library-query").focus();
  }
});
if (/Mac|iPhone|iPad/.test(navigator.platform)) $("focus-shortcut").textContent = "⌘ K";

updateStorageStatus();
navigate();
loadHealth();
loadLibrary();
