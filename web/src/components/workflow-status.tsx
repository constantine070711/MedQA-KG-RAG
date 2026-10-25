"use client";

import { Activity, Check, LoaderCircle } from "lucide-react";

const STEPS = [
  { key: "safety", label: "安全" },
  { key: "retrieval", label: "检索" },
  { key: "fusion", label: "融合" },
  { key: "generation", label: "生成" },
  { key: "analysis", label: "校验" },
] as const;

const STAGE_INDEX: Record<string, number> = {
  queued: 0,
  safety: 0,
  retrieval: 1,
  fusion: 2,
  generation: 3,
  analysis: 4,
  decision: 4,
  finalize: 4,
};

interface WorkflowStatusProps {
  stage?: string;
  label: string;
  detail?: string;
}

export function WorkflowStatus({
  stage,
  label,
  detail,
}: WorkflowStatusProps) {
  const activeIndex = STAGE_INDEX[stage ?? "queued"] ?? 0;

  return (
    <div
      className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--surface-raised)] px-3.5 py-3"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <span className="status-pulse mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-[var(--primary-soft)] text-[var(--primary)]">
          <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-[var(--ink)]">{label}</p>
          {detail ? (
            <p className="mt-0.5 text-xs leading-5 text-[var(--ink-soft)]">
              {detail}
            </p>
          ) : null}
          <div className="mt-3 grid grid-cols-5 gap-1.5" aria-hidden="true">
            {STEPS.map((step, index) => {
              const complete = index < activeIndex;
              const active = index === activeIndex;
              return (
                <div key={step.key} className="min-w-0">
                  <div
                    className={`h-1.5 rounded-full ${
                      complete
                        ? "bg-[var(--evidence)]"
                        : active
                          ? "bg-[var(--primary)]"
                          : "bg-[var(--line)]"
                    }`}
                  />
                  <div className="mt-1.5 flex items-center justify-center gap-1 text-[10px] text-[var(--ink-faint)]">
                    {complete ? (
                      <Check className="size-2.5" aria-hidden="true" />
                    ) : active ? (
                      <Activity className="size-2.5" aria-hidden="true" />
                    ) : null}
                    <span className="truncate">{step.label}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
