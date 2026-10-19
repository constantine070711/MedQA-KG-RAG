import type {
  AgentResponse,
  EvidenceItem,
  FusedContext,
  ReliabilityDecision,
  Triple,
} from "@/lib/types";

export const SAFETY_META = {
  safe: {
    label: "安全",
    description: "常规医学知识",
    tone: "safe",
  },
  caution: {
    label: "谨慎",
    description: "结合专业医生意见",
    tone: "caution",
  },
  restricted: {
    label: "受限",
    description: "不提供个人诊疗方案",
    tone: "restricted",
  },
  blocked: {
    label: "已拦截",
    description: "未生成实质医学内容",
    tone: "blocked",
  },
  unknown: {
    label: "待确认",
    description: "状态尚未生成",
    tone: "unknown",
  },
} as const;

export function safetyMeta(level?: string) {
  return SAFETY_META[(level ?? "unknown") as keyof typeof SAFETY_META] ??
    SAFETY_META.unknown;
}

export function decisionLabel(decision?: ReliabilityDecision) {
  switch (decision?.decision) {
    case "allow":
      return "允许回答";
    case "revise":
      return "附安全提示回答";
    case "refuse":
      return "拒绝实质回答";
    default:
      return "等待可靠性判断";
  }
}

export function asPercent(value?: number): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

export function evidenceTotal(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (value && typeof value === "object") {
    return Object.values(value as Record<string, unknown>).reduce<number>(
      (sum, item) => sum + (typeof item === "number" ? item : 0),
      0,
    );
  }
  return 0;
}

export function evidenceCounts(context?: FusedContext) {
  const sources = context?.evidence_sources ?? {};
  return {
    vector: evidenceTotal(sources.lightrag_vector),
    entities: evidenceTotal(sources.lightrag_graph),
    graph: evidenceTotal(sources.neo4j_graph),
  };
}

export function extractTriples(response?: AgentResponse): Triple[] {
  const items = response?.fused_context?.evidence ?? [];
  const triples: Triple[] = [];
  const seen = new Set<string>();

  for (const item of items) {
    if (item.kind !== "neo4j_triple") {
      continue;
    }
    const payload = item.payload ?? {};
    const source = String(
      payload.canonical_source ?? payload.source ?? "",
    ).trim();
    const relation = String(payload.relation_type ?? "").trim();
    const target = String(
      payload.canonical_target ?? payload.target ?? "",
    ).trim();
    if (!source || !target) {
      continue;
    }
    const key = `${source}|${relation}|${target}`;
    if (!seen.has(key)) {
      seen.add(key);
      triples.push({ source, relation, target });
    }
  }

  return triples;
}

export function evidenceItems(response?: AgentResponse): EvidenceItem[] {
  const items = (response?.fused_context?.evidence ?? []).filter(
    (item) => item && typeof item === "object",
  );
  const anchors = new Set(
    [
      ...(response?.mentions ?? []),
      ...(response?.citations?.reference_items ?? []).map(
        (item) => item.entity_name ?? "",
      ),
    ].filter(Boolean),
  );

  function relevance(item: EvidenceItem): number {
    if (item.kind === "neo4j_triple") {
      return 100;
    }
    const haystack = `${item.name ?? ""} ${evidenceDescription(item)}`;
    let score = 0;
    for (const anchor of anchors) {
      if (haystack.includes(anchor)) {
        score += 20;
      }
    }
    if (item.kind === "neo4j_node") {
      score += 8;
    }
    if (item.kind === "lightrag_entity") {
      score += 5;
    }
    return score;
  }

  return [...items].sort((left, right) => relevance(right) - relevance(left));
}

export function evidenceDescription(item: EvidenceItem): string {
  const payload = item.payload ?? {};
  switch (item.kind) {
    case "neo4j_triple":
      return `${String(payload.canonical_source ?? payload.source ?? "")} · ${
        String(payload.relation_type ?? "")
      } · ${String(payload.canonical_target ?? payload.target ?? "")}`;
    case "neo4j_node":
      return String(payload.name ?? item.name ?? "图谱节点");
    case "lightrag_chunk": {
      const content = String(payload.content ?? "");
      return content.length > 150 ? `${content.slice(0, 150)}…` : content;
    }
    default:
      return String(
        payload.description ?? item.name ?? "结构化知识证据",
      );
  }
}

export function evidenceKindLabel(kind?: string): string {
  const labels: Record<string, string> = {
    neo4j_triple: "图谱三元组",
    neo4j_node: "图谱实体",
    lightrag_entity: "实体证据",
    lightrag_relation: "关系证据",
    lightrag_chunk: "文档片段",
  };
  return labels[kind ?? ""] ?? "知识证据";
}

export function citationItems(response?: AgentResponse) {
  return response?.citations?.reference_items ?? [];
}

export function sourceLabels(response?: AgentResponse): string[] {
  const counts = evidenceCounts(response?.fused_context);
  const labels: string[] = [];
  if (counts.vector + counts.entities > 0) {
    labels.push("LightRAG");
  }
  if (counts.graph > 0) {
    labels.push("Neo4j");
  }
  return labels;
}
