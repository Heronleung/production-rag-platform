import assert from "node:assert/strict";
import test from "node:test";

import { SseParser } from "../lib/sse.ts";

const stream = [
  "event: citations\n",
  "data: {\"citations\":[{\"source\":\"a.pdf\"}]}\n\n",
  "event: token\n",
  "data: {\"text\":\"hello\"}\n\n",
  "event: done\n",
  "data: {\"elapsed_seconds\":1.2}\n\n",
].join("");

test("parses events even when every character is a separate network chunk", () => {
  const parser = new SseParser();
  const events = [...stream].flatMap((character) => parser.push(character));
  events.push(...parser.finish());
  assert.deepEqual(events.map((event) => event.event), ["citations", "token", "done"]);
  assert.equal(JSON.parse(events[1].data).text, "hello");
});

test("supports CRLF, comments, ids, and multiline data", () => {
  const parser = new SseParser();
  const events = parser.push(": keepalive\r\nid: 7\r\nevent: token\r\ndata: first\r\ndata: second\r\n\r\n");
  assert.deepEqual(events, [{ event: "token", data: "first\nsecond", id: "7" }]);
});

test("flushes a final event without a blank line", () => {
  const parser = new SseParser();
  parser.push("event: error\ndata: {\"detail\":\"stopped\"}");
  assert.equal(parser.finish()[0].event, "error");
});
