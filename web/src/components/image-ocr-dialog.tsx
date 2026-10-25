"use client";

import {
  ArrowRight,
  Check,
  FileImage,
  ImagePlus,
  LoaderCircle,
  ScanText,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";

import { API_BASE } from "@/lib/api";

type OcrWorker = Awaited<
  ReturnType<typeof import("tesseract.js")["createWorker"]>
>;

interface SupportedEntity {
  name: string;
  entity_type: string;
  aliases?: string[];
}

interface ImageOcrDialogProps {
  onClose: () => void;
  onUseText: (text: string) => void;
  onAskEntity: (entity: string) => void;
}

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_TEXT_LENGTH = 4000;

function normalizeForMatch(value: string): string {
  return value.toLocaleLowerCase("zh-CN").replace(/\s+/g, "");
}

function ocrStatusLabel(status: string): string {
  if (status.includes("loading tesseract core")) {
    return "加载 OCR 引擎";
  }
  if (status.includes("loading language")) {
    return "加载中文识别模型";
  }
  if (status.includes("initializing")) {
    return "初始化识别器";
  }
  if (status.includes("recognizing text")) {
    return "识别图片文字";
  }
  return "准备识别";
}

export function ImageOcrDialog({
  onClose,
  onUseText,
  onAskEntity,
}: ImageOcrDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [text, setText] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [entities, setEntities] = useState<SupportedEntity[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const workerRef = useRef<OcrWorker | null>(null);

  useEffect(() => {
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        void workerRef.current?.terminate();
        workerRef.current = null;
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      void workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, [previewUrl]);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/tools/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool: "list_supported_entities",
        arguments: { limit: 100 },
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("entity catalog unavailable");
        }
        return response.json();
      })
      .then((payload) => {
        const items = payload?.data?.entities;
        if (active && Array.isArray(items)) {
          setEntities(items);
        }
      })
      .catch(() => {
        if (active) {
          setEntities([]);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const detectedEntities = useMemo(() => {
    const normalized = normalizeForMatch(text);
    if (!normalized) {
      return [];
    }
    return entities
      .filter((entity) => {
        const candidates = [entity.name, ...(entity.aliases ?? [])];
        return candidates.some((candidate) => {
          const value = normalizeForMatch(candidate);
          return value.length >= 2 && normalized.includes(value);
        });
      })
      .slice(0, 8);
  }, [entities, text]);

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    event.target.value = "";
    if (!selected) {
      return;
    }
    if (!selected.type.startsWith("image/")) {
      setError("请选择 JPG、PNG 或 WebP 图片。");
      return;
    }
    if (selected.size > MAX_IMAGE_BYTES) {
      setError("图片不能超过 8 MB。");
      return;
    }

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setText("");
    setConfidence(null);
    setProgress(0);
    setStatus("");
    setError("");
  }

  async function runOcr() {
    if (!file || processing) {
      return;
    }
    setProcessing(true);
    setError("");
    setProgress(0);
    setStatus("加载 OCR 引擎");
    let worker: OcrWorker | null = null;

    try {
      const tesseract = await import("tesseract.js");
      worker = await tesseract.createWorker(["chi_sim", "eng"], 1, {
        logger: (message) => {
          setStatus(ocrStatusLabel(message.status));
          if (typeof message.progress === "number") {
            setProgress(Math.round(message.progress * 100));
          }
        },
      });
      workerRef.current = worker;
      await worker.setParameters({
        tessedit_pageseg_mode: tesseract.PSM.SPARSE_TEXT,
        preserve_interword_spaces: "1",
      });
      const result = await worker.recognize(file);
      const recognized = result.data.text.trim().slice(0, MAX_TEXT_LENGTH);
      setText(recognized);
      setConfidence(result.data.confidence);
      setStatus(recognized ? "识别完成" : "未识别到清晰文字");
      setProgress(100);
    } catch {
      setError(
        "图片识别失败。请确认网络可下载中文 OCR 模型，或换一张更清晰的图片。",
      );
      setStatus("");
    } finally {
      if (worker) {
        await worker.terminate().catch(() => undefined);
      }
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
      setProcessing(false);
    }
  }

  function closeDialog() {
    void workerRef.current?.terminate();
    workerRef.current = null;
    onClose();
  }

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-[rgba(12,28,25,0.46)] p-3 sm:p-6"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          closeDialog();
        }
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="image-ocr-title"
        className="flex max-h-[92dvh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface)] shadow-2xl"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[var(--line)] px-4 py-3 sm:px-5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-[var(--primary-soft)] text-[var(--primary)]">
            <ScanText className="size-5" aria-hidden="true" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 id="image-ocr-title" className="text-sm font-semibold">
              药盒 / 报告文字识别
            </h2>
            <p className="text-[10px] text-[var(--ink-faint)]">
              图片在浏览器本地识别，不自动发送到医疗问答服务
            </p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={closeDialog}
            title="关闭图片识别"
            aria-label="关闭图片识别"
            className="flex size-10 items-center justify-center rounded-lg text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
          >
            <X className="size-5" aria-hidden="true" />
          </button>
        </header>

        <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto p-4 sm:p-5">
          <div className="grid gap-4 md:grid-cols-[0.9fr_1.1fr]">
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="sr-only"
                onChange={selectFile}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={processing}
                className="flex min-h-64 w-full flex-col items-center justify-center overflow-hidden rounded-lg border border-dashed border-[var(--line-strong)] bg-[var(--surface-raised)] p-3 text-center transition hover:border-[var(--primary)] disabled:opacity-60"
              >
                {previewUrl ? (
                  // A local object URL is intentionally used instead of remote media.
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="待识别图片预览"
                    className="max-h-72 w-full object-contain"
                  />
                ) : (
                  <>
                    <span className="flex size-12 items-center justify-center rounded-full bg-[var(--primary-soft)] text-[var(--primary)]">
                      <ImagePlus className="size-6" aria-hidden="true" />
                    </span>
                    <span className="mt-3 text-xs font-semibold text-[var(--ink)]">
                      选择药盒、说明书或化验单图片
                    </span>
                    <span className="mt-1 text-[10px] text-[var(--ink-faint)]">
                      JPG / PNG / WebP，最大 8 MB
                    </span>
                  </>
                )}
              </button>

              {file ? (
                <div className="mt-2 flex items-center gap-2 text-[10px] text-[var(--ink-faint)]">
                  <FileImage className="size-3.5 shrink-0" aria-hidden="true" />
                  <span className="min-w-0 flex-1 truncate">{file.name}</span>
                  {confidence !== null ? (
                    <span className="shrink-0">
                      置信度 {Math.round(confidence)}%
                    </span>
                  ) : null}
                </div>
              ) : null}

              <button
                type="button"
                onClick={() => void runOcr()}
                disabled={!file || processing}
                className="mt-3 flex min-h-10 w-full items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-3 text-xs font-semibold text-white transition hover:bg-[var(--primary-strong)] disabled:bg-[var(--line)] disabled:text-[var(--ink-faint)]"
              >
                {processing ? (
                  <LoaderCircle
                    className="size-4 animate-spin"
                    aria-hidden="true"
                  />
                ) : (
                  <ScanText className="size-4" aria-hidden="true" />
                )}
                {processing ? "正在识别" : "识别图片文字"}
              </button>
            </div>

            <div className="flex min-h-64 flex-col">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="ocr-text"
                  className="text-xs font-semibold text-[var(--ink)]"
                >
                  识别文字
                </label>
                <span className="text-[10px] text-[var(--ink-faint)]">
                  {text.length}/{MAX_TEXT_LENGTH}
                </span>
              </div>
              <textarea
                id="ocr-text"
                value={text}
                onChange={(event) =>
                  setText(event.target.value.slice(0, MAX_TEXT_LENGTH))
                }
                placeholder="识别后可以在这里校对文字…"
                className="thin-scrollbar mt-2 min-h-56 flex-1 resize-y rounded-lg border border-[var(--line)] bg-[var(--surface-raised)] px-3 py-3 text-xs leading-6 text-[var(--ink)] outline-none placeholder:text-[var(--ink-faint)] focus:border-[var(--primary)]"
              />

              {processing || status ? (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-[10px] text-[var(--ink-faint)]">
                    <span>{status}</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[var(--line)]">
                    <div
                      className="h-full rounded-full bg-[var(--primary)] transition-[width] duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : null}

              {detectedEntities.length > 0 ? (
                <div className="mt-3">
                  <p className="text-[10px] font-semibold text-[var(--ink-faint)]">
                    已收录实体
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {detectedEntities.map((entity) => (
                      <button
                        key={`${entity.entity_type}-${entity.name}`}
                        type="button"
                        onClick={() => {
                          closeDialog();
                          onAskEntity(entity.name);
                        }}
                        className="flex min-h-8 items-center gap-1 rounded-full border border-[var(--line)] bg-[var(--surface-raised)] px-2.5 text-[10px] font-medium text-[var(--primary)] transition hover:border-[var(--primary)] hover:bg-[var(--primary-soft)]"
                        title={`询问 ${entity.name} 的医学知识`}
                      >
                        <Check className="size-3" aria-hidden="true" />
                        {entity.name}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              {error ? (
                <p
                  role="alert"
                  className="mt-3 rounded-lg bg-[var(--danger-soft)] px-3 py-2 text-[11px] leading-5 text-[var(--danger)]"
                >
                  {error}
                </p>
              ) : null}

              <div className="mt-3 flex items-start gap-2 rounded-lg bg-[var(--surface-muted)] px-3 py-2 text-[10px] leading-5 text-[var(--ink-soft)]">
                <ShieldCheck
                  className="mt-0.5 size-3.5 shrink-0 text-[var(--primary)]"
                  aria-hidden="true"
                />
                图片只在当前浏览器中识别。继续提问时，编辑后的文字会发送至问答服务；
                系统不会根据图片自动诊断或解释个人检查结果。
              </div>
            </div>
          </div>
        </div>

        <footer className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-t border-[var(--line)] px-4 py-3 sm:px-5">
          <button
            type="button"
            onClick={closeDialog}
            className="min-h-10 rounded-lg px-3 text-xs font-medium text-[var(--ink-soft)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--ink)]"
          >
            取消
          </button>
          <button
            type="button"
            onClick={() => {
              const value = text.trim().slice(0, 2000);
              if (!value) {
                return;
              }
              closeDialog();
              onUseText(value);
            }}
            disabled={!text.trim() || processing}
            className="flex min-h-10 items-center gap-1.5 rounded-lg bg-[var(--primary)] px-4 text-xs font-semibold text-white transition hover:bg-[var(--primary-strong)] disabled:bg-[var(--line)] disabled:text-[var(--ink-faint)]"
          >
            放入问题输入框
            <ArrowRight className="size-3.5" aria-hidden="true" />
          </button>
        </footer>
      </section>
    </div>
  );
}
