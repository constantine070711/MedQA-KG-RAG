"use client";

import { useMemo } from "react";
import type { Triple } from "@/lib/types";

interface EvidenceGraphProps {
  triples: Triple[];
}

interface GraphNode {
  id: string;
  x: number;
  y: number;
}

function truncate(value: string, length = 10) {
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

export function EvidenceGraph({ triples }: EvidenceGraphProps) {
  const graph = useMemo(() => {
    const nodeIds = Array.from(
      new Set(triples.flatMap((triple) => [triple.source, triple.target])),
    ).slice(0, 7);
    const centerX = 180;
    const centerY = 118;
    const radiusX = 126;
    const radiusY = 74;
    const nodes = new Map<string, GraphNode>();

    nodeIds.forEach((id, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(nodeIds.length, 1) - Math.PI / 2;
      nodes.set(id, {
        id,
        x: centerX + Math.cos(angle) * radiusX,
        y: centerY + Math.sin(angle) * radiusY,
      });
    });

    const edges = triples.filter(
      (triple) => nodes.has(triple.source) && nodes.has(triple.target),
    );
    return { nodes: Array.from(nodes.values()), edges };
  }, [triples]);

  if (triples.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-[var(--line-strong)] px-4 py-8 text-center text-sm text-[var(--ink-soft)]">
        本次回答没有产生 Neo4j 结构化关系。
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-raised)]">
      <svg
        viewBox="0 0 360 236"
        className="h-auto w-full"
        role="img"
        aria-label="Neo4j 知识图谱证据关系图"
      >
        <defs>
          <marker
            id="arrow-evidence"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#7aa79f" />
          </marker>
        </defs>
        {graph.edges.map((edge, index) => {
          const source = graph.nodes.find((node) => node.id === edge.source);
          const target = graph.nodes.find((node) => node.id === edge.target);
          if (!source || !target) {
            return null;
          }
          return (
            <g key={`${edge.source}-${edge.relation}-${edge.target}-${index}`}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke="#9abbb5"
                strokeWidth="1.4"
                markerEnd="url(#arrow-evidence)"
              />
              <title>{`${edge.source} → ${edge.relation} → ${edge.target}`}</title>
            </g>
          );
        })}
        {graph.nodes.map((node, index) => (
          <g key={node.id}>
            <circle
              cx={node.x}
              cy={node.y}
              r={index === 0 ? 23 : 20}
              fill={index === 0 ? "#0b726b" : "#ffffff"}
              stroke={index === 0 ? "#075a55" : "#84aaa3"}
              strokeWidth="1.5"
            >
              <title>{node.id}</title>
            </circle>
            <text
              x={node.x}
              y={node.y + 3.5}
              textAnchor="middle"
              fill={index === 0 ? "#ffffff" : "#26413d"}
              fontSize="9"
              fontWeight="650"
            >
              {truncate(node.id)}
              <title>{node.id}</title>
            </text>
          </g>
        ))}
      </svg>
      <div className="border-t border-[var(--line)] px-3 py-2 text-[11px] text-[var(--ink-faint)]">
        节点按证据关系生成，仅展示当前回答命中的子图。
      </div>
    </div>
  );
}
