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
const state = { view: "", topic: "", sources: null, health: null, libraryError: false, requests: {} };

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
  const tokens = /§([^§\n]+)§|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|\[(\d+)\]/g;
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
  const fence = /^\s*§§§[\w+-]*\s*$/;
  const unordered = /^\s*[-*+]\s+(.+)/;
  const ordered = /^\s*\d+\.\s+(.+)/;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    if (fence.test(line)) {
      const code = [];
      while (++i < lines.length && !/^\s*§§§\s*$/.test(lines[i])) code.push(lines[i]);
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

function navigate() {
  const [name, filter] = location.hash.slice(1).split("/");
  if (name === "main-content" && state.view) return;
  const view = Object.hasOwn(labels, name) ? name : "assistant";
  const topic = view === "library" && Object.hasOwn(topics, filter) ? filter : "";
  const changed = state.view !== view;
  state.view = view;
  state.topic = topic;
  $("page-label").textContent = labels[view];
  document.title = "DocLens · " + (view === "assistant" ? "AI Engineering Assistant" : labels[view]);
  document.querySelectorAll("[data-view]").forEach((link) => {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-topic]").forEach((link) => {
    link.classList.toggle("is-active", view === "library" && link.dataset.topic === topic);
  });
  for (const key of Object.keys(labels)) $("view-" + key).hidden = key !== view;
  const hadOpenMenu = $("sidebar").classList.contains("is-open");
  setMenu(false, false);
  if (view === "library") renderLibrary();
  if (changed) window.scrollTo(0, 0);
  if (hadOpenMenu) $("main-content").focus();
}

function resetAssistant() {
  const active = state.requests.ask;
  if (active) {
    delete state.requests.ask;
    active.abort();
  }
  $("ask-form").reset();
  $("ask-submit").disabled = false;
  $("ask-submit").removeAttribute("aria-busy");
  $("assistant-hero").hidden = false;
  $("suggestions").hidden = false;
  $("ask-output").hidden = true;
  $("ask-output").replaceChildren();
  location.hash = "assistant";
  navigate();
  window.scrollTo(0, 0);
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
  panel.append(element("h3", "", "Sources cited"));
  for (const citation of citations) {
    const link = externalLink(citation.url, "citation-link");
    const copy = element("span", "citation-copy");
    copy.append(element("strong", "", citation.title), element("small", "", citation.heading.at(-1) || citation.version));
    link.append(element("span", "citation-number", String(citation.number)), copy, icon("arrow-up-right"));
    panel.append(link);
  }
  return panel;
}

function answerResult(root, data, query, topic) {
  const title = element("h2", "question-title", query);
  const layout = element("div", "answer-layout");
  const card = element("article", "answer-card");
  const heading = element("div", "answer-head");
  const answerLabel = element("div", "answer-label");
  const name = element("div");
  name.append(
    element("h3", "", data.abstained ? "Insufficient evidence" : "DocLens answer"),
    element("p", "answer-mode", generationNames[data.generation] || data.generation),
  );
  answerLabel.append(topicIcon("", data.abstained ? "alert" : "sparkles"), name);
  heading.append(answerLabel, copyButton(data.answer));
  const content = element("div", "answer-body");
  content.append(markdown(data.answer, new Map(data.citations.map((cite) => [cite.number, cite]))));
  card.append(heading, content);
  if (data.note) card.append(element("p", "notice", data.note));
  layout.append(card);
  if (data.citations.length) layout.append(citationPanel(data.citations));
  root.replaceChildren(title, layout);
  if (data.evidence.length) {
    const detail = element("details", "answer-evidence");
    const openLab = element("button", "link-button", "Open in retrieval lab");
    openLab.type = "button";
    openLab.append(icon("arrow-up-right"));
    openLab.addEventListener("click", () => {
      $("search-question").setCustomValidity("");
      $("search-question").value = query;
      $("search-topic").value = topic;
      $("search-mode").value = data.mode;
      location.hash = "retrieval";
      navigate();
      $("search-form").requestSubmit();
    });
    detail.append(element("summary", "", "Inspect retrieved passages · " + data.evidence.length), openLab, passageList(data.evidence));
    root.append(detail);
  }
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

async function submit(kind, event) {
  event.preventDefault();
  const form = event.currentTarget;
  const input = $(kind + "-question");
  const question = input.value.trim();
  if (question.length < 2) {
    input.setCustomValidity("Enter at least two characters.");
    input.reportValidity();
    return;
  }
  const values = new FormData(form);
  const topic = values.get("topic");
  const body = { question, topic: topic || null };
  if (kind === "search") {
    body.mode = values.get("mode");
    body.top_k = Number(values.get("top_k"));
  }
  state.requests[kind]?.abort();
  const controller = new AbortController();
  state.requests[kind] = controller;

  const root = $(kind + "-output");
  const button = $(kind + "-submit");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  root.hidden = false;
  if (kind === "ask") {
    $("assistant-hero").hidden = true;
    $("suggestions").hidden = true;
  }
  root.replaceChildren(message("Searching the documentation", "Finding passages in the indexed sources…", "loading"));
  if (kind === "ask") window.scrollTo(0, 0);
  try {
    const result = await jsonRequest("/" + kind, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (state.requests[kind] !== controller) return;
    if (kind === "ask") answerResult(root, result, question, topic);
    else searchResult(root, result);
  } catch (error) {
    if (error.name === "AbortError" || state.requests[kind] !== controller) return;
    const failure = message("Couldn’t complete the request", "Check that the API is running, then try again.", "error");
    const retry = element("button", "link-button", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => form.requestSubmit());
    failure.lastElementChild.append(retry);
    root.replaceChildren(failure);
  } finally {
    if (state.requests[kind] === controller) {
      delete state.requests[kind];
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
    for (const key of Object.keys(topics)) {
      const count = state.sources.documents.filter((document) => document.topic === key).length;
      document.querySelector('[data-count="' + key + '"]').textContent = count;
    }
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
$("new-question").addEventListener("click", resetAssistant);
$("help-open").addEventListener("click", () => {
  if (smallScreen.matches) setMenu(false);
  $("help-dialog").showModal();
});
$("ask-form").addEventListener("submit", (event) => submit("ask", event));
$("search-form").addEventListener("submit", (event) => submit("search", event));
$("library-query").addEventListener("input", renderLibrary);
for (const pair of ["ask", "search"]) {
  $(pair + "-question").addEventListener("input", (event) => event.target.setCustomValidity(""));
}
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    $("ask-question").setCustomValidity("");
    $("ask-question").value = button.dataset.question;
    $("ask-topic").value = "";
    $("ask-form").requestSubmit();
  });
});
$("ask-question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !$("ask-submit").disabled) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $("sidebar").classList.contains("is-open")) setMenu(false);
  if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey) && !$("help-dialog").open) {
    event.preventDefault();
    if ($("sidebar").classList.contains("is-open")) setMenu(false, false);
    if (state.view === "assistant") $("ask-question").focus();
    else if (state.view === "retrieval") $("search-question").focus();
    else $("library-query").focus();
  }
});
if (/Mac|iPhone|iPad/.test(navigator.platform)) $("focus-shortcut").textContent = "⌘ K";

navigate();
loadHealth();
loadLibrary();
