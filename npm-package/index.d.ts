/**
 * AgentDeploy - Zero-to-production AI agent deployment framework.
 * TypeScript type definitions.
 */

export declare const VERSION: string;

export declare class AgentDeployError extends Error {
  constructor(message: string);
}

export interface ExporterOptions {
  redactionLevel?: "none" | "basic" | "strict";
  format?: "json" | "jsonl";
}

export interface SessionEvent {
  timestamp: number;
  event_type: string;
  data: Record<string, any>;
  metadata?: Record<string, any>;
  duration_ms?: number;
}

export interface SessionExportData {
  version: string;
  session_id: string;
  exported_at: number;
  events: SessionEvent[];
  metadata: Record<string, any>;
  total_events: number;
}

export interface ReplayResult {
  matched: number;
  diverged: number;
  divergences: Array<{
    index: number;
    expected: Record<string, any>;
    actual: Record<string, any>;
  }>;
}

export declare class SessionExporter {
  constructor(options?: ExporterOptions);
  exportSession(
    sessionId: string,
    events: SessionEvent[],
    metadata?: Record<string, any>
  ): SessionExportData;
}

export declare class SessionImporter {
  loadFromJSON(jsonString: string): SessionExportData;
  replay(
    session: SessionExportData,
    handler?: (event: SessionEvent) => Record<string, any> | null
  ): ReplayResult;
}
