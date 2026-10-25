"use client";

import {
  Activity,
  Bot,
  Check,
  ChevronRight,
  CircleAlert,
  Copy,
  Database,
  FileText,
  ImagePlus,
  Mic,
  MicOff,
  Network,
  PanelRightOpen,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Square,
  Stethoscope,
  Trash2,
  UserRound,
  Volume2,
  VolumeX,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { EvidencePanel } from "@/components/evidence-panel";
import { ImageOcrDialog } from "@/components/image-ocr-dialog";
import { WorkflowStatus } from "@/components/workflow-status";
import { useSpeechInput, useSpeechOutput } from "@/hooks/use-speech";
import { API_BASE, streamChat } from "@/lib/api";
import { decisionLabel, safetyMeta, sourceLabels } from "@/lib/format";
import type {
  ChatMessage,
  HistoryTurn,
  RetrievalMode,
  StreamEvent,
} from "@/lib/types";

const EXAMPLES = [
  {
    question: "2型糖尿病有哪些常见症状？",
    keyPoint: "基础事实",
  },
  {
    question: "二甲双胍用于治疗什么疾病？",
    keyPoint: "药物与适应证",
  },
  {
    question: "高血压有哪些常见并发症？",
    keyPoint: "疾病关系",
  },
  {
    question: "高血压和脑卒中有什么关系？",
    keyPoint: "图谱关联",
  },
] as const;

type ApiStatus = "checking" | "online" | "offline";

interface StageInfo {
  stage: string;
  label: string;
  detail?: string;
}

function createId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildHistory(messages: ChatMessage[]): HistoryTurn[] {
  const history: HistoryTurn[] = [];
  let lastQuestion = "";

  for (const message of messages) {
    if (message.role === "user") {
      lastQuestion = message.content;
      continue;
    }
    if (!lastQuestion || !message.response?.answer) {
      continue;
    }
    history.push({
      question: lastQuestion,
      answer: message.response.answer,
      safety_level: message.response.safety_level,
      is_final_refusal:
        message.response.reliability_decision?.is_final_refusal ?? false,
    });
    lastQuestion = "";
  }

  return history.slice(-6);
}

function ApiStatusBadge({ status }: { status: ApiStatus }) {
  const meta = {
    checking: {
      label: "连接中",
      className: "bg-[var(--surface-muted)] text-[var(--ink-soft)]",
      dot: "bg-[var(--ink-faint)]",
    },
    online: {
      label: "服务就绪",
      className: "bg-[var(--evidence-soft)] text-[var(--evidence)]",
      dot: "bg-[var(--evidence)]",
    },
    offline: {
      label: "API 未连接",
      className: "bg-[var(--danger-soft)] text-[var(--danger)]",
      dot: "bg-[var(--danger)]",
    },
  }[status];

  return (
    <span
      className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-medium sm:flex ${meta.className}`}
    >
      <span className={`size-1.5 rounded-full ${meta.dot}`} aria-hidden="true" />
      {meta.label}
    </span>
  );
}

export function MedQAWorkspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [stage, setStage] = useState<StageInfo | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [mode, setMode] = useState<RetrievalMode>("mix");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const speechInput = useSpeechInput((text) => setInput(text));
  const speechOutput = useSpeechOutput(() => setSpeakingMessageId(null));

  const latestResponse = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === "assistant" && messages[index].response) {
        return messages[index].response;
      }
    }
    return undefined;
  }, [messages]);

  const latestTools = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role === "assistant" && message.response) {
        return message.tools ?? [];
      }
    }
    return [];
  }, [messages]);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("health check failed");
        }
        return response.json();
      })
      .then(() => {
        if (active) {
          setApiStatus("online");
        }
      })
      .catch(() => {
        if (active) {
          setApiStatus("offline");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: isStreaming ? "auto" : "smooth",
      block: "end",
    });
  }, [messages, stage, isStreaming]);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const timer = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);

  function updateAssistant(
    assistantId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId ? updater(message) : message,
      ),
    );
  }

  async function handleSubmit(questionOverride?: string) {
    const question = (questionOverride ?? input).trim();
    if (!question || isStreaming) {
      return;
    }

    const history = buildHistory(messages);
    speechInput.stop();
    speechOutput.stop();
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: question,
    };
    const assistantId = createId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setIsStreaming(true);
    setStage({
      stage: "queued",
      label: "正在连接问答工作流",
      detail: "准备执行安全检查与证据检索",
    });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        { question, mode, history },
        (event: StreamEvent) => {
          if (event.type === "status") {
            setStage({
              stage: event.stage,
              label: event.label,
              detail: event.detail,
            });
            setApiStatus("online");
            return;
          }

          if (event.type === "answer_delta") {
            updateAssistant(assistantId, (message) => ({
              ...message,
              content: `${message.content}${event.delta}`,
            }));
            return;
          }

          if (event.type === "tool") {
            updateAssistant(assistantId, (message) => ({
              ...message,
              tools: [
                ...(message.tools ?? []),
                {
                  tool: event.tool,
                  label: event.label,
                  status: event.status,
                  summary: event.summary,
                  elapsed_ms: event.elapsed_ms,
                },
              ],
            }));
            return;
          }

          if (event.type === "final") {
            updateAssistant(assistantId, (message) => ({
              ...message,
              content: event.response.answer ?? message.content,
              response: event.response,
              streaming: false,
            }));
            setStage(null);
            return;
          }

          updateAssistant(assistantId, (message) => ({
            ...message,
            streaming: false,
            error: event.message,
          }));
          setStage(null);
        },
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        updateAssistant(assistantId, (message) => ({
          ...message,
          streaming: false,
          content: message.content || "本次生成已停止。",
        }));
      } else {
        setApiStatus("offline");
        updateAssistant(assistantId, (message) => ({
          ...message,
          streaming: false,
          error: "本次请求暂时无法完成，请确认 API 服务已启动。",
        }));
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsStreaming(false);
      setStage(null);
      window.setTimeout(() => textareaRef.current?.focus(), 0);
    }
  }

  function stopStreaming() {
    abortRef.current?.abort();
  }

  function resetConversation() {
    abortRef.current?.abort();
    speechInput.stop();
    speechOutput.stop();
    setMessages([]);
    setStage(null);
    setIsStreaming(false);
    setInput("");
    setDrawerOpen(false);
    setImageDialogOpen(false);
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  }

  async function copyAnswer(answer?: string) {
    if (!answer) {
      return;
    }
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden">
      <header className="relative z-20 flex h-16 shrink-0 items-center border-b border-[var(--line)] bg-[color-mix(in_srgb,var(--surface)_94%,transparent)] px-3 backdrop-blur sm:px-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-[var(--primary)] text-white shadow-sm">
            <Stethoscope className="size-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="truncate text-sm font-[750] tracking-normal">
                MedQA
              </p>
              <span className="rounded bg-[var(--primary-soft)] px-1.5 py-0.5 text-[9px] font-bold text-[var(--primary-strong)]">
                V2
              </span>
            </div>
            <p className="hidden text-[10px] text-[var(--ink-faint)] sm:block">
              医疗知识证据工作台
            </p>
          </div>
        </div>

        <div className="ml-4 hidden items-center gap-2 md:flex">
          <div className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--surface-raised)] p-1">
            {(["mix", "local", "global"] as const).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setMode(item)}
                disabled={isStreaming}
                className={`min-h-7 rounded-md px-2.5 text-[10px] font-semibold transition ${
                  mode === item
                    ? "bg-[var(--surface)] text-[var(--primary)] shadow-sm"
                    : "text-[var(--ink-faint)] hover:text-[var(--ink)]"
                }`}
                title={`检索模式：${item}`}
              >
                {item === "mix" ? "混合" : item === "local" ? "局部" : "全局"}
              </button>
            ))}
          </div>
          <ApiStatusBadge status={apiStatus} />
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          <button
            type="button"
            onClick={resetConversation}
            title="新建会话"
            aria-label="新建会话"
            className="flex size-10 items-center justify-center rounded-lg text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
          >
            <Plus className="size-5" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            disabled={!latestResponse}
            title="打开证据检查器"
            aria-label="打开证据检查器"
            className="flex size-10 items-center justify-center rounded-lg text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)] disabled:opacity-35 xl:hidden"
          >
            <PanelRightOpen className="size-5" aria-hidden="true" />
          </button>
        </div>
      </header>

      <main className="grid min-h-0 flex-1 lg:grid-cols-[236px_minmax(0,1fr)] xl:grid-cols-[236px_minmax(0,1fr)_400px]">
        <aside className="hidden min-h-0 flex-col border-r border-[var(--line)] bg-[color-mix(in_srgb,var(--surface)_78%,transparent)] lg:flex">
          <div className="flex min-h-0 flex-1 flex-col p-3">
            <button
              type="button"
              onClick={resetConversation}
              className="flex min-h-10 w-full items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-3 text-xs font-semibold text-white transition hover:bg-[var(--primary-strong)]"
            >
              <Plus className="size-4" aria-hidden="true" />
              新建问题
            </button>

            <div className="mt-6 flex items-center justify-between">
              <p className="text-[10px] font-semibold uppercase tracking-normal text-[var(--ink-faint)]">
                问题示例
              </p>
            </div>
            <div className="mt-2 space-y-1.5">
              {EXAMPLES.map((example) => (
                <button
                  key={example.question}
                  type="button"
                  disabled={isStreaming}
                  onClick={() => handleSubmit(example.question)}
                  className="group flex min-h-10 w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)] disabled:opacity-45"
                >
                  <Search
                    className="size-3.5 shrink-0 text-[var(--ink-faint)] group-hover:text-[var(--primary)]"
                    aria-hidden="true"
                  />
                  <span className="line-clamp-2 min-w-0 flex-1 text-[11px] leading-4">
                    {example.question}
                  </span>
                  <ChevronRight
                    className="size-3.5 shrink-0 opacity-0 transition group-hover:opacity-100"
                    aria-hidden="true"
                  />
                </button>
              ))}
            </div>

            <div className="mt-auto space-y-2 pt-6">
              {messages.length > 0 ? (
                <button
                  type="button"
                  onClick={resetConversation}
                  className="flex min-h-10 w-full items-center gap-2 rounded-lg px-2.5 text-xs text-[var(--ink-soft)] transition hover:bg-[var(--danger-soft)] hover:text-[var(--danger)]"
                >
                  <Trash2 className="size-4" aria-hidden="true" />
                  清空当前会话
                </button>
              ) : null}
              <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-raised)] px-3 py-2.5">
                <div className="flex items-center justify-between text-[10px] text-[var(--ink-faint)]">
                  <span>当前版本</span>
                  <span className="font-semibold text-[var(--primary)]">
                    v2.0 dev
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-1.5 text-[10px] text-[var(--ink-faint)]">
                  <ShieldCheck className="size-3" aria-hidden="true" />
                  v1.0.0 已保留
                </div>
              </div>
            </div>
          </div>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-col">
          <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 sm:px-6">
              {messages.length === 0 ? (
                <div className="flex flex-1 flex-col justify-center py-10 sm:py-16">
                  <div className="max-w-2xl">
                    <div className="flex items-center gap-2 text-[var(--primary)]">
                      <span className="flex size-8 items-center justify-center rounded-lg bg-[var(--primary-soft)]">
                        <Activity className="size-4" aria-hidden="true" />
                      </span>
                      <span className="text-xs font-semibold">MedQA v2</span>
                    </div>
                    <h1 className="mt-5 max-w-xl text-2xl font-[780] leading-tight tracking-normal text-[var(--ink)] sm:text-3xl">
                      输入一个医疗知识问题
                    </h1>
                    <p className="mt-3 max-w-xl text-sm leading-6 text-[var(--ink-soft)]">
                      系统将结合文档检索与结构化图谱证据生成回答。
                    </p>
                  </div>

                  <div className="mt-8 grid gap-2 sm:grid-cols-2">
                    {EXAMPLES.map((example, index) => (
                      <button
                        key={example.question}
                        type="button"
                        onClick={() => handleSubmit(example.question)}
                        disabled={isStreaming}
                        className="group min-h-20 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 text-left transition hover:-translate-y-0.5 hover:border-[var(--line-strong)] hover:shadow-sm disabled:opacity-50"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[10px] font-semibold text-[var(--primary)]">
                            0{index + 1}
                          </span>
                          <ChevronRight
                            className="size-4 text-[var(--ink-faint)] transition group-hover:translate-x-0.5 group-hover:text-[var(--primary)]"
                            aria-hidden="true"
                          />
                        </div>
                        <p className="mt-3 text-sm font-semibold leading-6 text-[var(--ink)]">
                          {example.question}
                        </p>
                        <p className="mt-1 text-[10px] text-[var(--ink-faint)]">
                          {example.keyPoint}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="flex-1 space-y-8 py-6 sm:py-8" aria-live="polite">
                  {messages.map((message) =>
                    message.role === "user" ? (
                      <article
                        key={message.id}
                        className="message-enter flex justify-end gap-3"
                      >
                        <div className="max-w-[86%] rounded-2xl rounded-tr-md bg-[var(--primary)] px-4 py-3 text-sm leading-6 text-white shadow-sm">
                          {message.content}
                        </div>
                        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--surface)] text-[var(--ink-soft)]">
                          <UserRound className="size-4" aria-hidden="true" />
                        </div>
                      </article>
                    ) : (
                      <AssistantMessage
                        key={message.id}
                        message={message}
                        stage={stage}
                        isStreaming={isStreaming}
                        onCopy={() =>
                          copyAnswer(
                            message.response?.answer || message.content,
                          )
                        }
                        copied={copied}
                        onOpenEvidence={() => setDrawerOpen(true)}
                        speechSupported={speechOutput.supported}
                        speechPlaying={speakingMessageId === message.id}
                        onToggleSpeech={() => {
                          const answer =
                            message.response?.answer || message.content;
                          if (speakingMessageId === message.id) {
                            speechOutput.stop();
                            setSpeakingMessageId(null);
                          } else if (answer) {
                            speechOutput.speak(answer);
                            setSpeakingMessageId(message.id);
                          }
                        }}
                      />
                    ),
                  )}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>
          </div>

          <div className="shrink-0 border-t border-[var(--line)] bg-[color-mix(in_srgb,var(--surface)_92%,transparent)] px-4 pb-3 pt-3 backdrop-blur sm:px-6">
            <form
              className="mx-auto w-full max-w-3xl"
              onSubmit={(event) => {
                event.preventDefault();
                void handleSubmit();
              }}
            >
              <div className="flex items-end gap-2 rounded-xl border border-[var(--line-strong)] bg-[var(--surface)] p-2 shadow-sm transition focus-within:border-[var(--primary)] focus-within:ring-3 focus-within:ring-[color-mix(in_srgb,var(--primary)_12%,transparent)]">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      void handleSubmit();
                    }
                  }}
                  rows={1}
                  maxLength={2000}
                  placeholder="请输入医疗知识问题…"
                  aria-label="医疗知识问题"
                  className="thin-scrollbar max-h-36 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm leading-6 text-[var(--ink)] outline-none placeholder:text-[var(--ink-faint)]"
                />
                <button
                  type="button"
                  onClick={() => setImageDialogOpen(true)}
                  disabled={isStreaming}
                  title="识别药盒或化验单图片"
                  aria-label="识别药盒或化验单图片"
                  className="flex size-10 shrink-0 items-center justify-center rounded-lg text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)] disabled:opacity-35"
                >
                  <ImagePlus className="size-4" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  onClick={speechInput.toggle}
                  disabled={!speechInput.supported || isStreaming}
                  title={
                    speechInput.supported
                      ? speechInput.listening
                        ? "停止语音输入"
                        : "语音输入"
                      : "当前浏览器不支持语音输入"
                  }
                  aria-label={
                    speechInput.listening ? "停止语音输入" : "语音输入"
                  }
                  aria-pressed={speechInput.listening}
                  className={`flex size-10 shrink-0 items-center justify-center rounded-lg transition disabled:opacity-35 ${
                    speechInput.listening
                      ? "bg-[var(--danger-soft)] text-[var(--danger)]"
                      : "text-[var(--ink-soft)] hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
                  }`}
                >
                  {speechInput.listening ? (
                    <MicOff className="size-4" aria-hidden="true" />
                  ) : (
                    <Mic className="size-4" aria-hidden="true" />
                  )}
                </button>
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={stopStreaming}
                    title="停止生成"
                    aria-label="停止生成"
                    className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--ink)] transition hover:bg-[var(--line)]"
                  >
                    <Square className="size-4 fill-current" aria-hidden="true" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    title="发送问题"
                    aria-label="发送问题"
                    className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--primary)] text-white transition hover:bg-[var(--primary-strong)] disabled:bg-[var(--line)] disabled:text-[var(--ink-faint)]"
                  >
                    <Send className="size-4" aria-hidden="true" />
                  </button>
                )}
              </div>
              {speechInput.error ? (
                <p
                  role="alert"
                  className="mt-2 text-center text-[10px] text-[var(--danger)]"
                >
                  {speechInput.error}
                </p>
              ) : null}
              <p className="mt-2 text-center text-[10px] leading-4 text-[var(--ink-faint)]">
                医疗知识辅助，不替代医生诊断或处方。
              </p>
            </form>
          </div>
        </section>

        <aside className="hidden min-h-0 border-l border-[var(--line)] xl:block">
          <EvidencePanel response={latestResponse} tools={latestTools} />
        </aside>
      </main>

      {drawerOpen ? (
        <div className="xl:hidden">
          <button
            type="button"
            aria-label="关闭证据检查器"
            title="关闭证据检查器"
            className="fixed inset-0 z-40 bg-[rgba(15,34,31,0.28)]"
            onClick={() => setDrawerOpen(false)}
          />
          <EvidencePanel
            response={latestResponse}
            tools={latestTools}
            drawer
            onClose={() => setDrawerOpen(false)}
          />
        </div>
      ) : null}

      {imageDialogOpen ? (
        <ImageOcrDialog
          onClose={() => setImageDialogOpen(false)}
          onUseText={(text) => {
            setInput(text);
            window.setTimeout(() => textareaRef.current?.focus(), 0);
          }}
          onAskEntity={(entity) => {
            void handleSubmit(`请介绍${entity}相关的医学知识。`);
          }}
        />
      ) : null}
    </div>
  );
}

interface AssistantMessageProps {
  message: ChatMessage;
  stage: StageInfo | null;
  isStreaming: boolean;
  copied: boolean;
  onCopy: () => void;
  onOpenEvidence: () => void;
  speechSupported: boolean;
  speechPlaying: boolean;
  onToggleSpeech: () => void;
}

function AssistantMessage({
  message,
  stage,
  isStreaming,
  copied,
  onCopy,
  onOpenEvidence,
  speechSupported,
  speechPlaying,
  onToggleSpeech,
}: AssistantMessageProps) {
  const safety = safetyMeta(message.response?.safety_level);
  const sources = sourceLabels(message.response);
  const decision = message.response?.reliability_decision;
  const toolCount = message.tools?.length ?? 0;

  return (
    <article className="message-enter flex gap-3">
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-[var(--primary-soft)] text-[var(--primary)]">
        <Bot className="size-4" aria-hidden="true" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-semibold text-[var(--ink)]">MedQA</p>
          {message.response ? (
            <>
              <span
                className={`rounded-full px-2 py-0.5 text-[9px] font-semibold ${
                  safety.tone === "safe"
                    ? "bg-[var(--evidence-soft)] text-[var(--evidence)]"
                    : safety.tone === "blocked"
                      ? "bg-[var(--danger-soft)] text-[var(--danger)]"
                      : "bg-[var(--caution-soft)] text-[var(--caution)]"
                }`}
              >
                {safety.label}
              </span>
              <span className="text-[10px] text-[var(--ink-faint)]">
                {decisionLabel(decision)}
              </span>
            </>
          ) : null}
        </div>

        {message.content ? (
          <div
            className={`prose-medical mt-2 ${
              message.streaming ? "stream-caret" : ""
            }`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        ) : message.streaming && stage ? (
          <WorkflowStatus
            stage={stage.stage}
            label={stage.label}
            detail={stage.detail}
          />
        ) : null}

        {message.error ? (
          <div
            role="alert"
            className="mt-3 flex items-start gap-2 rounded-lg bg-[var(--danger-soft)] px-3 py-2 text-xs leading-5 text-[var(--danger)]"
          >
            <CircleAlert
              className="mt-0.5 size-4 shrink-0"
              aria-hidden="true"
            />
            <span>{message.error}</span>
          </div>
        ) : null}

        {message.response && !message.streaming ? (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {sources.map((source) => (
              <span
                key={source}
                className="flex items-center gap-1 rounded-full border border-[var(--line)] bg-[var(--surface-raised)] px-2 py-1 text-[9px] font-medium text-[var(--ink-soft)]"
              >
                {source === "Neo4j" ? (
                  <Network className="size-2.5" aria-hidden="true" />
                ) : (
                  <FileText className="size-2.5" aria-hidden="true" />
                )}
                {source}
              </span>
            ))}
            {toolCount > 0 ? (
              <span className="flex items-center gap-1 rounded-full border border-[var(--line)] bg-[var(--surface-raised)] px-2 py-1 text-[9px] font-medium text-[var(--ink-soft)]">
                <Wrench className="size-2.5" aria-hidden="true" />
                {toolCount} 个工具
              </span>
            ) : null}
            <button
              type="button"
              onClick={onToggleSpeech}
              disabled={!speechSupported}
              title={
                speechSupported
                  ? speechPlaying
                    ? "停止朗读"
                    : "朗读回答"
                  : "当前浏览器不支持语音朗读"
              }
              aria-label={speechPlaying ? "停止朗读" : "朗读回答"}
              aria-pressed={speechPlaying}
              className="flex min-h-8 items-center gap-1 rounded-md px-2 text-[10px] text-[var(--ink-faint)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)] disabled:opacity-35"
            >
              {speechPlaying ? (
                <VolumeX className="size-3" aria-hidden="true" />
              ) : (
                <Volume2 className="size-3" aria-hidden="true" />
              )}
              {speechPlaying ? "停止" : "朗读"}
            </button>
            <button
              type="button"
              onClick={onCopy}
              title="复制回答"
              aria-label="复制回答"
              className="flex min-h-8 items-center gap-1 rounded-md px-2 text-[10px] text-[var(--ink-faint)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
            >
              {copied ? (
                <Check className="size-3" aria-hidden="true" />
              ) : (
                <Copy className="size-3" aria-hidden="true" />
              )}
              {copied ? "已复制" : "复制"}
            </button>
            <button
              type="button"
              onClick={onOpenEvidence}
              className="flex min-h-8 items-center gap-1 rounded-md px-2 text-[10px] text-[var(--primary)] transition hover:bg-[var(--primary-soft)] xl:hidden"
            >
              <Database className="size-3" aria-hidden="true" />
              查看证据
            </button>
          </div>
        ) : null}

        {isStreaming && message.streaming && message.content && stage ? (
          <p className="mt-3 flex items-center gap-1.5 text-[10px] text-[var(--ink-faint)]">
            <span className="size-1.5 rounded-full bg-[var(--primary)] status-pulse" />
            {stage.label}
          </p>
        ) : null}
      </div>
    </article>
  );
}
