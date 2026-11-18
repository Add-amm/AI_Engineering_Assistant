const PREFIX = "doclens.chat.v1.";
const CHAT_ID = /^[a-z0-9-]+$/i;

function readChat(raw, id) {
  try {
    const chat = JSON.parse(raw);
    if (chat?.id !== id || typeof chat.title !== "string" || !Number.isFinite(chat.updatedAt)) return null;
    if (!Array.isArray(chat.turns) || !chat.turns.length) return null;
    return chat.turns.every((turn) => turn && typeof turn.id === "string" && typeof turn.question === "string" &&
      (turn.result === null || (typeof turn.result?.answer === "string" &&
        Array.isArray(turn.result.citations) && Array.isArray(turn.result.evidence)))) ? chat : null;
  } catch {
    return null;
  }
}

export function loadChats() {
  const chats = new Map();
  try {
    for (const key of Object.keys(localStorage)) {
      if (!key.startsWith(PREFIX)) continue;
      const id = key.slice(PREFIX.length);
      if (!CHAT_ID.test(id)) continue;
      const chat = readChat(localStorage.getItem(key), id);
      if (chat) chats.set(id, chat);
    }
    return { chats, available: true };
  } catch {
    return { chats, available: false };
  }
}

export function saveChat(chat) {
  localStorage.setItem(PREFIX + chat.id, JSON.stringify(chat));
}

export function removeChat(id) {
  localStorage.removeItem(PREFIX + id);
}
