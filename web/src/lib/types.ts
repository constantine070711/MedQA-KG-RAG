export type SafetyLevel =
  | "safe"
  | "caution"
  | "restricted"
  | "blocked"
  | "unknown";

export type RetrievalMode = "naive" | "local" | "global" | "hybrid" | "mix";

export interface EvidenceItem {
  kind?: string;
  name?: string;
  sources?: string[];
  priority?: number;
  payload?: Record<string, unknown>;
}

export interface FusedContext {
  status?: string;
  evidence?: EvidenceItem[];
  evidence_sources?: Record<string, number | Record<string, number>>;
  coverage?: {
    notes?: string[];
    [key: string]: unknown;
  };
}

export interface ReliabilityDecision {
  decision?: string;
  is_final_refusal?: boolean;
  confidence?: number;
  reliability_score?: number;
  reason_code?: string;
  reason?: string;
  triggered_rules?: string[];
  uncertainty_level?: string;
  citation_quality?: number;
}

export interface CitationItem {
  raw_text?: string;
  entity_name?: string;
  source_file?: string;
  entity_type?: string;
}

export interface CitationAnalysis {
  citation_present?: boolean;
  reference_items?: CitationItem[];
  grounded_entities?: string[];
  ungrounded_entities?: string[];
  grounded_ratio?: number;
  total_references?: number;
  grounding_method?: string;
}

export interface AgentError {
  type?: string;
  message?: string;
}

export interface AgentResponse {
  question?: string;
  answer?: string;
  status?: string;
  safety_level?: SafetyLevel | string;
  intent?: string;
  mentions?: string[];
  neo4j_unavailable?: boolean;
  llm_generation_count?: number;
  reliability_decision?: ReliabilityDecision;
  citations?: CitationAnalysis;
  fused_context?: FusedContext;
  coverage?: Record<string, unknown>;
  errors?: AgentError[];
}

export interface HistoryTurn {
  question: string;
  answer: string;
  safety_level?: string;
  is_final_refusal?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: AgentResponse;
  streaming?: boolean;
  error?: string;
}

export type StreamEvent =
  | {
      type: "status";
      stage: string;
      label: string;
      detail?: string;
      request_id?: string;
    }
  | {
      type: "answer_delta";
      delta: string;
    }
  | {
      type: "final";
      request_id: string;
      elapsed_ms: number;
      response: AgentResponse;
    }
  | {
      type: "error";
      message: string;
      error_type?: string;
      request_id?: string;
    };

export interface Triple {
  source: string;
  relation: string;
  target: string;
}
