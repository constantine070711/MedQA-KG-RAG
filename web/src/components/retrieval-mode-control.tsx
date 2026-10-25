"use client";

import { CircleHelp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { RetrievalMode } from "@/lib/types";

const RETRIEVAL_MODES = ["mix", "local", "global"] as const;

const RETRIEVAL_MODE_GUIDE = [
  {
    mode: "mix",
    label: "混合",
    description: "综合具体实体、全局关系与文本片段，适合多数问题。",
    recommended: true,
  },
  {
    mode: "local",
    label: "局部",
    description: "聚焦问题中提到的具体疾病、药物或症状。",
    recommended: false,
  },
  {
    mode: "global",
    label: "全局",
    description: "综合知识库中的主题和整体关系，适合概括性问题。",
    recommended: false,
  },
] as const;

export function RetrievalModeControl({
  value,
  onChange,
  disabled,
}: {
  value: RetrievalMode;
  onChange: (mode: RetrievalMode) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const currentMode =
    RETRIEVAL_MODE_GUIDE.find((item) => item.mode === value) ??
    RETRIEVAL_MODE_GUIDE[0];

  useEffect(() => {
    if (!open) {
      return;
    }

    function closeOnOutsidePointer(event: PointerEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
        setPinned(false);
      }
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
        setPinned(false);
      }
    }

    document.addEventListener("pointerdown", closeOnOutsidePointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  function close() {
    setOpen(false);
    setPinned(false);
  }

  function toggle() {
    if (pinned) {
      close();
      return;
    }
    setPinned(true);
    setOpen(true);
  }

  return (
    <div className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--surface-raised)] p-1">
      <div className="hidden items-center gap-1 md:flex">
        {RETRIEVAL_MODES.map((mode) => {
          const guide = RETRIEVAL_MODE_GUIDE.find((item) => item.mode === mode);
          return (
            <button
              key={mode}
              type="button"
              onClick={() => onChange(mode)}
              disabled={disabled}
              className={`min-h-7 rounded-md px-2.5 text-[10px] font-semibold transition ${
                value === mode
                  ? "bg-[var(--surface)] text-[var(--primary)] shadow-sm"
                  : "text-[var(--ink-faint)] hover:text-[var(--ink)]"
              }`}
              title={`检索模式：${mode}。${guide?.description ?? ""}`}
            >
              {guide?.label ?? mode}
            </button>
          );
        })}
      </div>

      <span
        className="mx-0.5 hidden h-4 w-px bg-[var(--line)] md:block"
        aria-hidden="true"
      />

      <div
        ref={containerRef}
        className="relative flex items-center"
        onMouseEnter={() => {
          if (!pinned) {
            setOpen(true);
          }
        }}
        onMouseLeave={() => {
          if (!pinned) {
            setOpen(false);
          }
        }}
      >
        <button
          type="button"
          onClick={toggle}
          onFocus={() => setOpen(true)}
          disabled={disabled}
          aria-label={`检索模式：${currentMode.label}，查看说明`}
          aria-expanded={open}
          aria-controls="retrieval-mode-help"
          className="flex min-h-7 items-center justify-center gap-1 rounded-md px-2 text-[10px] font-semibold text-[var(--ink-faint)] transition hover:bg-[var(--surface)] hover:text-[var(--primary)] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--primary)] disabled:opacity-40 md:size-7 md:px-0"
        >
          <span className="md:hidden">{currentMode.label}</span>
          <CircleHelp className="size-3.5" aria-hidden="true" />
        </button>

        {open ? (
          <div
            id="retrieval-mode-help"
            role="dialog"
            aria-labelledby="retrieval-mode-help-title"
            className="absolute left-1/2 top-[calc(100%+0.55rem)] z-30 w-[min(21rem,calc(100vw-1.5rem))] -translate-x-1/2 rounded-xl border border-[var(--line-strong)] bg-[var(--surface-raised)] p-3 text-left shadow-xl md:left-auto md:right-0 md:translate-x-0"
          >
            <div className="mb-2.5 flex items-start justify-between gap-3">
              <div>
                <p
                  id="retrieval-mode-help-title"
                  className="text-xs font-bold text-[var(--ink)]"
                >
                  检索模式
                </p>
                <p className="mt-0.5 text-[10px] leading-4 text-[var(--ink-faint)]">
                  选择系统查找医学证据的范围和方式。
                </p>
              </div>
              <button
                type="button"
                onClick={close}
                aria-label="关闭检索模式说明"
                className="flex size-6 shrink-0 items-center justify-center rounded-md text-[var(--ink-faint)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
              >
                <X className="size-3.5" aria-hidden="true" />
              </button>
            </div>

            <div className="space-y-2">
              {RETRIEVAL_MODE_GUIDE.map((item) => (
                <button
                  key={item.mode}
                  type="button"
                  onClick={() => {
                    onChange(item.mode);
                    close();
                  }}
                  disabled={disabled}
                  aria-pressed={item.mode === value}
                  className={`w-full rounded-lg border px-2.5 py-2 text-left transition disabled:opacity-50 ${
                    item.mode === value
                      ? "border-[var(--primary)] bg-[var(--primary-soft)]"
                      : "border-[var(--line)] bg-[var(--surface)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-muted)]"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-bold text-[var(--ink)]">
                      {item.label}
                    </span>
                    {item.mode === value ? (
                      <span className="text-[8px] font-semibold text-[var(--primary)]">
                        当前
                      </span>
                    ) : null}
                    {item.recommended ? (
                      <span className="rounded bg-[var(--evidence-soft)] px-1.5 py-0.5 text-[8px] font-bold text-[var(--evidence)]">
                        推荐
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-1 text-[10px] leading-4 text-[var(--ink-soft)]">
                    {item.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
