import type { HistoryTurn, RetrievalMode, StreamEvent } from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_MEDQA_API_URL ?? "http://localhost:8000";

interface ChatPayload {
  question: string;
  mode: RetrievalMode;
  history: HistoryTurn[];
}

export async function streamChat(
  payload: ChatPayload,
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Streaming response is unavailable in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");

      if (!data) {
        continue;
      }

      try {
        onEvent(JSON.parse(data) as StreamEvent);
      } catch {
        // Ignore malformed SSE frames and keep the stream alive.
      }
    }

    if (done) {
      break;
    }
  }
}
