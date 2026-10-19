"use client";

import { useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  CircleAlert,
  Database,
  FileText,
  Network,
  Quote,
  ShieldCheck,
  X,
} from "lucide-react";

import { EvidenceGraph } from "@/components/evidence-graph";
import {
  asPercent,
  citationItems,
  decisionLabel,
  evidenceCounts,
  evidenceDescription,
  evidenceItems,
  evidenceKindLabel,
  extractTriples,
  safetyMeta,
  sourceLabels,
} from "@/lib/format";
import type { AgentResponse } from "@/lib/types";

type EvidenceTab = "evidence" | "graph" | "citations";

interface EvidencePanelProps {
  response?: AgentResponse;
  drawer?: boolean;
  onClose?: () => void;
}

function EmptyEvidence() {
  return (
    <div className="flex h-full min-h-80 flex-col items-center justify-center px-6 text-center">
      <div className="flex size-12 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--surface-raised)] text-[var(--primary)]">
        <BookOpen className="size-5" aria-hidden="true" />
      </div>
      <p className="mt-4 text-sm font-semibold text-[var(--ink)]">
        回答后在这里核查依据
      </p>
      <p className="mt-1 max-w-56 text-xs leading-5 text-[var(--ink-soft)]">
        系统会展示检索来源、图谱关系、可靠性判断与引用追溯结果。
      </p>
    </div>
  );
}

export function EvidencePanel({
  response,
  drawer = false,
  onClose,
}: EvidencePanelProps) {
  const [tab, setTab] = useState<EvidenceTab>("evidence");
  const safety = safetyMeta(response?.safety_level);
  const counts = evidenceCounts(response?.fused_context);
  const triples = extractTriples(response);
  const items = evidenceItems(response);
  const citations = citationItems(response);
  const sources = sourceLabels(response);
  const grounded = asPercent(response?.citations?.grounded_ratio);
  const confidence = asPercent(
    response?.reliability_decision?.confidence ??
      response?.reliability_decision?.reliability_score,
  );
  const decision = response?.reliability_decision;

  return (
    <div
      className={
        drawer
          ? "fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-[var(--line)] bg-[var(--surface)] shadow-2xl"
          : "flex h-full min-h-0 flex-col bg-[var(--surface)]"
      }
    >
      <div className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--line)] px-4">
        <div className="flex items-center gap-2.5">
          <span className="flex size-8 items-center justify-center rounded-lg bg-[var(--evidence-soft)] text-[var(--evidence)]">
            <ShieldCheck className="size-4" aria-hidden="true" />
          </span>
          <div>
            <p className="text-sm font-semibold">证据检查器</p>
            <p className="text-[11px] text-[var(--ink-faint)]">
              安全、可靠性、来源与图谱
            </p>
          </div>
        </div>
        {drawer ? (
          <button
            type="button"
            onClick={onClose}
            title="关闭证据检查器"
            aria-label="关闭证据检查器"
            className="flex size-10 items-center justify-center rounded-lg text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
          >
            <X className="size-5" aria-hidden="true" />
          </button>
        ) : null}
      </div>

      {!response ? (
        <EmptyEvidence />
      ) : (
        <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto">
          <div className="border-b border-[var(--line)] p-4">
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-lg bg-[var(--surface-muted)] px-3 py-2.5">
                <p className="text-[10px] uppercase tracking-normal text-[var(--ink-faint)]">
                  安全等级
                </p>
                <p className="mt-1 text-sm font-semibold">{safety.label}</p>
              </div>
              <div className="rounded-lg bg-[var(--surface-muted)] px-3 py-2.5">
                <p className="text-[10px] text-[var(--ink-faint)]">可靠性</p>
                <p className="mt-1 text-sm font-semibold">{confidence}</p>
              </div>
              <div className="rounded-lg bg-[var(--surface-muted)] px-3 py-2.5">
                <p className="text-[10px] text-[var(--ink-faint)]">引用追溯</p>
                <p className="mt-1 text-sm font-semibold">{grounded}</p>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-[var(--line)] px-3 py-2.5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <CheckCircle2
                    className="size-4 shrink-0 text-[var(--evidence)]"
                    aria-hidden="true"
                  />
                  <p className="truncate text-xs font-semibold">
                    {decisionLabel(decision)}
                  </p>
                </div>
                <span className="shrink-0 text-[10px] text-[var(--ink-faint)]">
                  {response.llm_generation_count ?? 0} 次生成
                </span>
              </div>
              {decision?.reason ? (
                <p className="mt-1.5 line-clamp-2 text-[11px] leading-5 text-[var(--ink-soft)]">
                  {decision.reason}
                </p>
              ) : null}
            </div>

            {safety.tone !== "safe" ? (
              <div
                className={`mt-3 flex items-start gap-2 rounded-lg px-3 py-2.5 text-xs leading-5 ${
                  safety.tone === "blocked"
                    ? "bg-[var(--danger-soft)] text-[var(--danger)]"
                    : "bg-[var(--caution-soft)] text-[var(--caution)]"
                }`}
              >
                <CircleAlert
                  className="mt-0.5 size-4 shrink-0"
                  aria-hidden="true"
                />
                <span>{safety.description}</span>
              </div>
            ) : null}

            <div className="mt-3 flex flex-wrap gap-1.5">
              {sources.length > 0 ? (
                sources.map((source) => (
                  <span
                    key={source}
                    className="rounded-full border border-[var(--line)] bg-[var(--surface-raised)] px-2 py-1 text-[10px] font-medium text-[var(--ink-soft)]"
                  >
                    {source}
                  </span>
                ))
              ) : (
                <span className="text-[11px] text-[var(--ink-faint)]">
                  暂无结构化来源
                </span>
              )}
            </div>
          </div>

          <div className="sticky top-0 z-10 border-b border-[var(--line)] bg-[var(--surface)] px-3 py-2">
            <div
              className="grid grid-cols-3 gap-1 rounded-lg bg-[var(--surface-muted)] p-1"
              role="tablist"
              aria-label="证据检查器视图"
            >
              <button
                type="button"
                role="tab"
                onClick={() => setTab("evidence")}
                className={`flex min-h-9 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-medium transition ${
                  tab === "evidence"
                    ? "bg-[var(--surface)] text-[var(--ink)] shadow-sm"
                    : "text-[var(--ink-soft)] hover:text-[var(--ink)]"
                }`}
                aria-selected={tab === "evidence"}
              >
                <FileText className="size-3.5" aria-hidden="true" />
                证据
              </button>
              <button
                type="button"
                role="tab"
                onClick={() => setTab("graph")}
                className={`flex min-h-9 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-medium transition ${
                  tab === "graph"
                    ? "bg-[var(--surface)] text-[var(--ink)] shadow-sm"
                    : "text-[var(--ink-soft)] hover:text-[var(--ink)]"
                }`}
                aria-selected={tab === "graph"}
              >
                <Network className="size-3.5" aria-hidden="true" />
                图谱
              </button>
              <button
                type="button"
                role="tab"
                onClick={() => setTab("citations")}
                className={`flex min-h-9 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-medium transition ${
                  tab === "citations"
                    ? "bg-[var(--surface)] text-[var(--ink)] shadow-sm"
                    : "text-[var(--ink-soft)] hover:text-[var(--ink)]"
                }`}
                aria-selected={tab === "citations"}
              >
                <Quote className="size-3.5" aria-hidden="true" />
                引用
              </button>
            </div>
          </div>

          <div className="p-4">
            {tab === "evidence" ? (
              <div className="space-y-2.5">
                {items.length > 0 ? (
                  items.slice(0, 8).map((item, index) => (
                    <article
                      key={`${item.kind}-${item.name}-${index}`}
                      className="rounded-xl border border-[var(--line)] bg-[var(--surface-raised)] p-3"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-semibold text-[var(--primary)]">
                          {evidenceKindLabel(item.kind)}
                        </span>
                        <span className="text-[10px] text-[var(--ink-faint)]">
                          P{item.priority ?? index + 1}
                        </span>
                      </div>
                      <p className="mt-1.5 text-xs font-medium leading-5 text-[var(--ink)]">
                        {item.name || "结构化知识证据"}
                      </p>
                      <p className="mt-1 line-clamp-3 text-[11px] leading-5 text-[var(--ink-soft)]">
                        {evidenceDescription(item)}
                      </p>
                      {item.sources?.length ? (
                        <p className="mt-2 text-[10px] text-[var(--ink-faint)]">
                          来源：{item.sources.join(" · ")}
                        </p>
                      ) : null}
                    </article>
                  ))
                ) : (
                  <div className="rounded-xl border border-dashed border-[var(--line-strong)] px-4 py-8 text-center text-xs text-[var(--ink-soft)]">
                    当前回答未产生可展示的结构化证据。
                  </div>
                )}
              </div>
            ) : null}

            {tab === "graph" ? (
              <div className="space-y-3">
                <EvidenceGraph triples={triples} />
                {triples.length > 0 ? (
                  <div className="space-y-2">
                    {triples.slice(0, 5).map((triple, index) => (
                      <div
                        key={`${triple.source}-${triple.target}-${index}`}
                        className="rounded-lg border border-[var(--line)] px-3 py-2 text-[11px] leading-5"
                      >
                        <span className="font-semibold">{triple.source}</span>
                        <span className="mx-1.5 text-[var(--primary)]">
                          → {triple.relation} →
                        </span>
                        <span className="font-semibold">{triple.target}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            {tab === "citations" ? (
              <div className="space-y-2.5">
                {citations.length > 0 ? (
                  citations.map((citation, index) => (
                    <article
                      key={`${citation.entity_name}-${index}`}
                      className="rounded-xl border border-[var(--line)] bg-[var(--surface-raised)] p-3"
                    >
                      <div className="flex items-start gap-2">
                        <Quote
                          className="mt-0.5 size-3.5 shrink-0 text-[var(--primary)]"
                          aria-hidden="true"
                        />
                        <div className="min-w-0">
                          <p className="text-xs font-semibold">
                            {citation.entity_name ||
                              citation.raw_text ||
                              "知识引用"}
                          </p>
                          {citation.raw_text &&
                          citation.raw_text !== citation.entity_name ? (
                            <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-[var(--ink-soft)]">
                              {citation.raw_text}
                            </p>
                          ) : null}
                          {citation.source_file ? (
                            <p className="mt-1.5 break-all text-[10px] text-[var(--ink-faint)]">
                              {citation.source_file}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    </article>
                  ))
                ) : (
                  <div className="rounded-xl border border-dashed border-[var(--line-strong)] px-4 py-8 text-center text-xs text-[var(--ink-soft)]">
                    当前回答没有生成可展示的引用条目。
                  </div>
                )}
              </div>
            ) : null}

            <div className="mt-4 rounded-xl border border-[var(--line)] bg-[var(--surface-muted)] p-3">
              <div className="flex items-center gap-2 text-xs font-semibold">
                <Database className="size-4 text-[var(--primary)]" aria-hidden="true" />
                检索覆盖
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-base font-semibold">{counts.vector}</p>
                  <p className="text-[10px] text-[var(--ink-faint)]">向量片段</p>
                </div>
                <div>
                  <p className="text-base font-semibold">{counts.entities}</p>
                  <p className="text-[10px] text-[var(--ink-faint)]">图证据</p>
                </div>
                <div>
                  <p className="text-base font-semibold">{counts.graph}</p>
                  <p className="text-[10px] text-[var(--ink-faint)]">图谱命中</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
