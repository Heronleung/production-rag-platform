export type SseEvent = {
  event: string;
  data: string;
  id?: string;
};

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  let id: string | undefined;
  const data: string[] = [];

  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") id = value;
  }

  if (data.length === 0) return null;
  return { event, data: data.join("\n"), ...(id === undefined ? {} : { id }) };
}

export class SseParser {
  private buffer = "";

  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const events: SseEvent[] = [];
    let match: RegExpExecArray | null;
    const boundary = /\r?\n\r?\n/;

    while ((match = boundary.exec(this.buffer)) !== null) {
      const block = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
    }
    return events;
  }

  finish(): SseEvent[] {
    const tail = this.buffer.trim();
    this.buffer = "";
    if (!tail) return [];
    const parsed = parseBlock(tail);
    return parsed ? [parsed] : [];
  }
}

export async function consumeSse(
  response: Response,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("The response did not contain a stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const event of parser.push(decoder.decode(value, { stream: true }))) {
        onEvent(event);
      }
    }
    for (const event of parser.push(decoder.decode())) onEvent(event);
    for (const event of parser.finish()) onEvent(event);
  } finally {
    reader.releaseLock();
  }
}
