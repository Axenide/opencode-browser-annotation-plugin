import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { Plugin as V2Plugin } from "@opencode/plugin";
import type { Plugin as V1Plugin, PluginInput } from "@opencode-ai/plugin";

/**
 * opencode-browser-annotation-plugin
 *
 * Runs a loopback HTTP server on the OpenCode host. A browser extension POSTs
 * annotations (a typed instruction plus selected-element metadata) to it. When
 * the browser runs on a separate desktop, the extension reaches this server over
 * an `ssh -L` local forward (desktop -> host).
 *
 * On receipt the plugin injects a new user turn into the chosen OpenCode session
 * (the extension may target any session by id; otherwise the most recently
 * active one) so the agent responds.
 *
 * Scope: text + element metadata only. No screenshots, no image/vision.
 */

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 39_517;

/**
 * Each OpenCode process runs its own in-process API server, and a plugin's
 * `client` can only see the sessions of ITS OWN process. To list/target sessions
 * across every running OpenCode (different projects and processes), each plugin
 * instance also starts a tiny "peer" HTTP server on an ephemeral port and drops
 * a registry file naming it. Whichever instance wins the shared endpoint port
 * (DEFAULT_PORT) fans /status and /annotations out to every registered peer and
 * merges the results, so the browser extension sees one unified session list and
 * an annotation reaches whichever process owns the target session.
 */
function registryDir(): string {
  const base = process.env.XDG_DATA_HOME?.trim() || join(homedir(), ".local", "share");
  const dir = join(base, "opencode", "annotation-peers");
  try {
    mkdirSync(dir, { recursive: true });
    return dir;
  } catch {
    const fallback = join(tmpdir(), "opencode-annotation-peers");
    mkdirSync(fallback, { recursive: true });
    return fallback;
  }
}

interface PeerRecord {
  pid: number;
  port: number;
  directory: string;
  updated: number;
}

interface ElementMeta {
  selector?: string;
  tag?: string;
  id?: string;
  name?: string;
  testId?: string;
  role?: string;
  ariaLabel?: string;
  classes?: string[];
  text?: string;
  href?: string;
  src?: string;
  bounds?: { x: number; y: number; width: number; height: number };
  inShadow?: boolean;
  inIframe?: boolean;
  framePath?: string;
  ancestors?: string[];
  landmark?: string;
  componentPath?: string;
  framework?: string;
  html?: string;
}

interface Annotation {
  instruction?: string;
  page?: { url?: string; title?: string };
  element?: ElementMeta;
}

interface SubmitPayload {
  annotations?: Annotation[];
  sessionID?: string;
  extensionVersion?: string;
}

function envHost(): string {
  return process.env.OPENCODE_ANNOTATION_HOST?.trim() || DEFAULT_HOST;
}

function envPort(): number {
  const raw = process.env.OPENCODE_ANNOTATION_PORT?.trim();
  const n = raw ? Number.parseInt(raw, 10) : DEFAULT_PORT;
  return Number.isFinite(n) && n > 0 && n < 65_536 ? n : DEFAULT_PORT;
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

/**
 * Surface the most code-locatable identifiers first: test ids and framework
 * hooks are what map to source, not pixel bounds.
 */
function formatElement(el: ElementMeta | undefined): string {
  if (!el) return "  (no element captured)";
  const lines: string[] = [];
  const tag = el.tag ? el.tag.toLowerCase() : "element";
  lines.push(`  Element: <${tag}>`);
  if (el.testId) lines.push(`  data-testid: ${el.testId}`);
  if (el.id) lines.push(`  id: ${el.id}`);
  if (el.name) lines.push(`  name: ${el.name}`);
  if (el.role) lines.push(`  role: ${el.role}`);
  if (el.ariaLabel) lines.push(`  aria-label: ${el.ariaLabel}`);
  if (el.classes && el.classes.length) lines.push(`  classes: ${el.classes.slice(0, 8).join(" ")}`);
  if (el.href) lines.push(`  href: ${el.href}`);
  if (el.src) lines.push(`  src: ${el.src}`);
  if (el.text) lines.push(`  text: ${JSON.stringify(truncate(el.text.trim(), 200))}`);
  if (el.componentPath) lines.push(`  ${el.framework ?? "component"} components: ${el.componentPath}`);
  if (el.landmark) lines.push(`  nearest region: ${el.landmark}`);
  if (el.ancestors && el.ancestors.length) lines.push(`  ancestors (nearest first): ${el.ancestors.join(" < ")}`);
  if (el.selector) lines.push(`  css path: ${el.selector}`);
  const context: string[] = [];
  if (el.inShadow) context.push("inside a shadow DOM");
  if (el.inIframe) context.push(`inside an iframe${el.framePath ? ` (${el.framePath})` : ""}`);
  if (context.length) lines.push(`  context: ${context.join(", ")}`);
  if (el.bounds) {
    const b = el.bounds;
    lines.push(`  viewport bounds: ${Math.round(b.width)}×${Math.round(b.height)} at (${Math.round(b.x)}, ${Math.round(b.y)})`);
  }
  if (el.html) lines.push(`  opening tag: ${truncate(el.html.trim(), 500)}`);
  return lines.join("\n");
}

function annotationBlock(a: Annotation, index: number, total: number): string {
  const label = total > 1 ? ` ${index + 1}` : "";
  const page = a.page ?? {};
  const instruction = (a.instruction ?? "").trim() || "(no instruction text)";
  return [
    `### Annotation${label}`,
    `Instruction: ${instruction}`,
    page.url ? `Page: ${page.title ? `${page.title} — ` : ""}${page.url}` : "",
    "Selected element:",
    formatElement(a.element),
  ]
    .filter(Boolean)
    .join("\n");
}

function buildPrompt(annotations: Annotation[]): string {
  // The static interpretation rules live in the `browser-annotation` skill, not
  // in every payload. The header marks this as a browser annotation and tells
  // the agent to use that skill, so the rules are stated once (in the skill),
  // never repeated per message.
  const header =
    annotations.length > 1
      ? `Browser annotations (${annotations.length}) — handle using the \`browser-annotation\` skill. Address each one.`
      : "Browser annotation — handle using the `browser-annotation` skill.";

  const blocks = annotations.map((a, i) => annotationBlock(a, i, annotations.length));

  return [header, "", ...blocks].join("\n");
}

async function readJsonBody(req: IncomingMessage, limitBytes = 2_000_000): Promise<unknown> {
  return await new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > limitBytes) {
        reject(new Error("payload too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (error) {
        reject(error instanceof Error ? error : new Error("invalid JSON"));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  // Set headers individually and assign statusCode before end(). Combined
  // writeHead(status, headers) is not reliably flushed by the Bun node:http
  // compatibility layer that OpenCode runs under, which yields empty replies.
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  // The extension calls from a browser origin; allow it to read the response.
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "content-type");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.statusCode = status;
  res.end(payload);
}

interface SessionInfo {
  id: string;
  title: string;
  updated: number;
  directory?: string;
}

interface RuntimeServices {
  directory: string;
  listSessions(): Promise<SessionInfo[]>;
  inject(sessionID: string, prompt: string): Promise<void>;
  log(level: "info" | "warn" | "error", message: string, extra?: Record<string, unknown>): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseSessions(value: unknown, directory: string): SessionInfo[] {
  const rows = Array.isArray(value)
    ? value
    : isRecord(value) && Array.isArray(value.data)
      ? value.data
      : [];

  return rows
    .filter((row): row is Record<string, unknown> => isRecord(row))
    .filter((row) => typeof row.id === "string" && !row.parentID)
    .map((row) => {
      const time = isRecord(row.time) ? row.time : {};
      const location = isRecord(row.location) ? row.location : {};
      return {
        id: row.id as string,
        title: typeof row.title === "string" ? row.title : (row.id as string),
        updated: typeof time.updated === "number" ? time.updated : 0,
        directory: typeof location.directory === "string" ? location.directory : directory,
      };
    });
}

function createRuntime(services: RuntimeServices) {
  const host = envHost();
  const port = envPort();
  const { directory, log } = services;
  const peerID = `${process.pid}-${randomUUID()}`;

  let activeSessionID: string | null = null;
  let server: Server | null = null; // the shared endpoint (DEFAULT_PORT), if we won it
  let peerServer: Server | null = null; // this instance's own peer server
  let peerPort = 0;
  let peerFile: string | null = null;
  let registryTimer: ReturnType<typeof setInterval> | null = null;
  let started = false;
  let stopped = false;

  const PEER_STALE_MS = 30 * 1000; // a peer file older than this is treated as dead
  const REGISTRY_REFRESH_MS = 10 * 1000;

  // Sessions this run has touched (created / messaged / status / idle). Used
  // only to bias the picker ordering — NOT to filter — so sessions from other
  // OpenCode processes and other project directories still appear.
  const activeIDs = new Set<string>();
  // The picker shows every recent session merged across all instances.
  const RECENT_MAX = 25;

  /** Sessions owned by THIS OpenCode process (its own in-process server). */
  async function localSessions(): Promise<SessionInfo[]> {
    try {
      return await services.listSessions();
    } catch (error) {
      log("warn", `session.list failed: ${error instanceof Error ? error.message : "unknown"}`);
      return [];
    }
  }

  /** Inject into a session THIS process owns. */
  async function localInject(sessionID: string, annotations: Annotation[]): Promise<{ ok: boolean; error?: string }> {
    try {
      await services.inject(sessionID, buildPrompt(annotations));
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "session.prompt failed" };
    }
  }

  // ——— Peer registry: discover other OpenCode processes' plugin instances ———

  function readPeers(): PeerRecord[] {
    const dir = registryDir();
    const now = Date.now();
    const out: PeerRecord[] = [];
    let files: string[] = [];
    try {
      files = readdirSync(dir).filter((f) => f.endsWith(".json"));
    } catch {
      return out;
    }
    for (const f of files) {
      const full = join(dir, f);
      try {
        const rec = JSON.parse(readFileSync(full, "utf8")) as PeerRecord;
        const alive = typeof rec.pid === "number" && isAlive(rec.pid);
        const fresh = typeof rec.updated === "number" && now - rec.updated < PEER_STALE_MS;
        if (rec.port && alive && fresh) out.push(rec);
        else if (!alive) rmSync(full, { force: true }); // reap dead instance's file
      } catch {
        rmSync(full, { force: true });
      }
    }
    return out;
  }

  function isAlive(pid: number): boolean {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  }

  async function peerFetch(port: number, path: string, body?: unknown): Promise<any | null> {
    try {
      const res = await fetch(`http://${host}:${port}${path}`, {
        method: body ? "POST" : "GET",
        headers: body ? { "content-type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(4000),
      });
      return await res.json();
    } catch {
      return null;
    }
  }

  /**
   * Merged picker list: this process's own sessions plus every live peer's,
   * de-duplicated, touched-this-run first, then newest. Only the endpoint owner
   * fans out; a plain peer just returns its own via /local.
   */
  async function mergedSessions(): Promise<SessionInfo[]> {
    const mine = await localSessions();
    const peers = readPeers().filter((p) => p.port !== peerPort);
    const peerLists = await Promise.all(
      peers.map(async (p) => {
        const data = await peerFetch(p.port, "/local");
        const rows = data && Array.isArray(data.sessions) ? (data.sessions as SessionInfo[]) : [];
        return rows;
      }),
    );

    const byId = new Map<string, SessionInfo>();
    for (const s of [...mine, ...peerLists.flat()]) {
      const prev = byId.get(s.id);
      if (!prev || s.updated > prev.updated) byId.set(s.id, s);
    }
    const all = [...byId.values()];
    const existing = new Set(all.map((s) => s.id));
    for (const id of activeIDs) if (!existing.has(id)) activeIDs.delete(id);

    const byRecent = all.sort((a, b) => b.updated - a.updated);
    const touched = byRecent.filter((s) => activeIDs.has(s.id));
    const rest = byRecent.filter((s) => !activeIDs.has(s.id));
    return [...touched, ...rest].slice(0, RECENT_MAX);
  }

  /** Route an inject to whichever instance owns the target session. */
  async function routeInject(
    sessionID: string,
    annotations: Annotation[],
  ): Promise<{ ok: boolean; error?: string }> {
    const mineIds = new Set((await localSessions()).map((s) => s.id));
    if (mineIds.has(sessionID)) return localInject(sessionID, annotations);

    for (const p of readPeers()) {
      if (p.port === peerPort) continue;
      const data = await peerFetch(p.port, "/local");
      const ids: string[] = data && Array.isArray(data.sessions) ? data.sessions.map((s: SessionInfo) => s.id) : [];
      if (ids.includes(sessionID)) {
        const r = await peerFetch(p.port, "/inject", { sessionID, annotations });
        if (r && r.ok) return { ok: true };
        return { ok: false, error: (r && r.error) || "peer inject failed" };
      }
    }
    // Not found on any peer; last-ditch try locally (id may have just moved).
    return localInject(sessionID, annotations);
  }

  async function handleSubmit(
    annotations: Annotation[],
    requestedSessionID?: string,
  ): Promise<{ ok: boolean; error?: string; injected: number; sessionID?: string }> {
    const targetID = requestedSessionID || activeSessionID;
    if (!targetID) {
      return {
        ok: false,
        error: "No target session. Send a message in OpenCode first, or pick a session.",
        injected: 0,
      };
    }
    if (requestedSessionID) {
      const sessions = await mergedSessions();
      if (!sessions.some((s) => s.id === requestedSessionID)) {
        return { ok: false, error: "Target session no longer exists.", injected: 0 };
      }
    }
    const result = await routeInject(targetID, annotations);
    if (!result.ok) return { ok: false, error: result.error, injected: 0, sessionID: targetID };
    return { ok: true, injected: annotations.length, sessionID: targetID };
  }

  // ——— Peer server: this instance's own endpoint for the fan-out owner ———

  function peerHandle(req: IncomingMessage, res: ServerResponse): void {
    if (req.method === "GET" && req.url === "/local") {
      void localSessions().then((sessions) => sendJson(res, 200, { ok: true, sessions }));
      return;
    }
    if (req.method === "POST" && req.url === "/inject") {
      readJsonBody(req)
        .then(async (parsed) => {
          const p = (parsed ?? {}) as SubmitPayload;
          const annotations = Array.isArray(p.annotations) ? p.annotations : [];
          if (!p.sessionID || annotations.length === 0) {
            sendJson(res, 400, { ok: false, error: "sessionID and annotations required." });
            return;
          }
          const r = await localInject(p.sessionID, annotations);
          sendJson(res, r.ok ? 200 : 409, r);
        })
        .catch((error: unknown) => sendJson(res, 400, { ok: false, error: error instanceof Error ? error.message : "bad request" }));
      return;
    }
    sendJson(res, 404, { ok: false, error: "Not found" });
  }

  function writePeerFile(): void {
    if (!peerPort) return;
    try {
      const dir = registryDir();
      peerFile = join(dir, `${peerID}.json`);
      const rec: PeerRecord = { pid: process.pid, port: peerPort, directory, updated: Date.now() };
      writeFileSync(peerFile, JSON.stringify(rec));
    } catch {
      /* registry unavailable — this instance simply won't be discoverable */
    }
  }

  function startPeerServer(): void {
    const s = createServer(peerHandle);
    peerServer = s;
    s.on("error", () => {
      /* ephemeral port clash is unlikely; if it happens, stay undiscoverable */
      if (peerServer === s) peerServer = null;
    });
    s.listen(0, host, () => {
      if (stopped) {
        s.close();
        return;
      }
      const addr = s.address();
      peerPort = typeof addr === "object" && addr ? addr.port : 0;
      writePeerFile();
      registryTimer = setInterval(writePeerFile, REGISTRY_REFRESH_MS);
      log("info", `Annotation peer server on http://${host}:${peerPort} (pid ${process.pid})`);
    });
  }

  function dispose(): void {
    if (stopped) return;
    stopped = true;
    if (registryTimer) clearInterval(registryTimer);
    registryTimer = null;
    if (peerFile) {
      try {
        rmSync(peerFile, { force: true });
      } catch {
        /* ignore */
      }
    }
    peerServer?.close();
    peerServer = null;
    server?.close();
    server = null;
  }

  // ——— Shared endpoint (the port the extension talks to) ———

  function handle(req: IncomingMessage, res: ServerResponse): void {
    if (req.method === "OPTIONS") {
      sendJson(res, 204, {});
      return;
    }
    if (req.method === "GET" && req.url === "/status") {
      void mergedSessions().then((sessions) => {
        const active = sessions.find((s) => s.id === activeSessionID);
        sendJson(res, 200, {
          ok: true,
          activeSession: Boolean(activeSessionID),
          sessionID: activeSessionID,
          sessionTitle: active?.title ?? null,
          sessions,
          host,
          port,
        });
      });
      return;
    }
    if (req.method === "POST" && req.url === "/annotations") {
      readJsonBody(req)
        .then(async (parsed) => {
          const payload = (parsed ?? {}) as SubmitPayload;
          const annotations = Array.isArray(payload.annotations) ? payload.annotations : [];
          if (annotations.length === 0) {
            sendJson(res, 400, { ok: false, error: "No annotations in payload." });
            return;
          }
          const result = await handleSubmit(annotations, payload.sessionID);
          if (result.ok) {
            log("info", `Injected ${result.injected} annotation(s)`, { sessionID: result.sessionID });
            sendJson(res, 200, result);
          } else {
            log("warn", `Annotation submit failed: ${result.error}`);
            sendJson(res, 409, result);
          }
        })
        .catch((error: unknown) => {
          const message = error instanceof Error ? error.message : "bad request";
          sendJson(res, 400, { ok: false, error: message });
        });
      return;
    }
    sendJson(res, 404, { ok: false, error: "Not found" });
  }

  function startEndpoint(): void {
    if (server) return; // never double-listen (the plugin may init more than once)
    const s = createServer(handle);
    server = s; // claim synchronously so a re-entrant start() is a no-op
    s.on("error", (error: NodeJS.ErrnoException) => {
      // Another instance already owns the endpoint. Drop this half-open server;
      // we still serve our sessions to the owner via the peer server.
      server = null;
      s.close();
      if (error.code !== "EADDRINUSE") {
        log("error", `Annotation server error: ${error.message}`);
      }
    });
    s.listen(port, host, () => {
      if (!stopped) log("info", `Browser annotation server listening on http://${host}:${port}`);
    });
  }

  function start(): void {
    if (started || stopped) return;
    started = true;
    startPeerServer();
    startEndpoint();
  }

  function onPrompt(sessionID: string | undefined): void {
    if (!sessionID) return;
    activeSessionID = sessionID;
    activeIDs.add(sessionID);
  }

  function onV1Event(value: unknown): void {
    if (!isRecord(value) || typeof value.type !== "string") return;
    const properties = isRecord(value.properties) ? value.properties : {};
    const info = isRecord(properties.info) ? properties.info : {};
    const id = typeof info.id === "string" ? info.id : properties.sessionID;
    if (typeof id !== "string") return;
    if (value.type === "session.deleted") {
      activeIDs.delete(id);
      if (id === activeSessionID) activeSessionID = null;
      return;
    }
    if (["session.created", "session.updated", "session.status", "session.idle"].includes(value.type)) {
      activeIDs.add(id);
    }
  }

  function onV2Event(value: unknown): void {
    if (!isRecord(value) || typeof value.type !== "string") return;
    const data = isRecord(value.data) ? value.data : {};
    const id = data.sessionID;
    if (typeof id !== "string") return;
    if (value.type === "session.deleted") {
      activeIDs.delete(id);
      if (id === activeSessionID) activeSessionID = null;
      return;
    }
    if (["session.created", "session.execution.started", "session.idle"].includes(value.type)) {
      activeIDs.add(id);
    }
  }

  return { start, dispose, onPrompt, onV1Event, onV2Event };
}

const legacyPlugin: V1Plugin = async ({ client, directory }: PluginInput) => {
  const runtime = createRuntime({
    directory,
    async listSessions() {
      return parseSessions(await client.session.list({}), directory);
    },
    async inject(sessionID, prompt) {
      await client.session.promptAsync({
        path: { id: sessionID },
        body: { parts: [{ type: "text", text: prompt }] },
      });
    },
    log(level, message, extra) {
      void client.app.log({ body: { service: "browser-annotation", level, message, extra } }).catch(() => {});
    },
  });

  try {
    runtime.start();
  } catch (error) {
    runtime.dispose();
    throw error;
  }

  return {
    "chat.message": async (input) => runtime.onPrompt(input.sessionID),
    event: async ({ event }) => runtime.onV1Event(event),
    dispose: async () => runtime.dispose(),
  };
};

const v2Plugin = V2Plugin.define({
  id: "opencode-browser-annotation-plugin",
  async setup(ctx) {
    const directory = ctx.location.directory;
    const runtime = createRuntime({
      directory,
      async listSessions() {
        const page = await ctx.session.list({ limit: 50, order: "desc", parentID: null, directory });
        return page.data.map((session) => ({
          id: session.id,
          title: session.title ?? session.id,
          updated: session.time.updated,
          directory: session.location.directory,
        }));
      },
      async inject(sessionID, prompt) {
        await ctx.session.prompt({ sessionID, text: prompt });
      },
      log(level, message, extra) {
        console[level]("[browser-annotation]", message, extra ?? {});
      },
    });

    const promptRegistration = await ctx.session.hook("prompt", (event) => runtime.onPrompt(event.sessionID));
    const controller = new AbortController();
    const eventTask = (async () => {
      try {
        for await (const event of ctx.event.subscribe({
          types: ["session.created", "session.deleted", "session.execution.started", "session.idle"],
          signal: controller.signal,
        })) {
          runtime.onV2Event(event);
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          console.error("[browser-annotation] Event subscription failed", error);
        }
      }
    })();

    try {
      runtime.start();
    } catch (error) {
      controller.abort();
      try {
        await promptRegistration.dispose();
      } finally {
        runtime.dispose();
      }
      await eventTask;
      throw error;
    }

    return async () => {
      controller.abort();
      try {
        await promptRegistration.dispose();
      } finally {
        runtime.dispose();
      }
      await eventTask;
    };
  },
});

export const BrowserAnnotationPlugin = {
  ...v2Plugin,
  server: legacyPlugin,
};

export default BrowserAnnotationPlugin;
