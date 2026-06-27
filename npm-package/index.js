/**
 * AgentDeploy - Zero-to-production AI agent deployment framework.
 *
 * This is the npm package for the AgentDeploy framework.
 * The primary implementation is in Python (pip install substrai-agentdeploy).
 * This package provides TypeScript type definitions and a lightweight runtime SDK.
 *
 * @see https://github.com/substrai/agentdeploy
 * @author Gaurav Kumar Sinha <gaurav@substrai.dev>
 */

"use strict";

const VERSION = "1.3.0";

class AgentDeployError extends Error {
  constructor(message) {
    super(message);
    this.name = "AgentDeployError";
  }
}

class SessionExporter {
  constructor(options = {}) {
    this.redactionLevel = options.redactionLevel || "basic";
    this.format = options.format || "json";
  }

  exportSession(sessionId, events, metadata = {}) {
    const exportData = {
      version: "1.0.0",
      session_id: sessionId,
      exported_at: Date.now() / 1000,
      events: events.map((e) => this._redactEvent(e)),
      metadata,
      total_events: events.length,
    };
    return exportData;
  }

  _redactEvent(event) {
    if (this.redactionLevel === "none") return event;
    const redacted = { ...event };
    const sensitiveKeys = ["api_key", "token", "secret", "password", "authorization"];
    if (redacted.data) {
      redacted.data = { ...redacted.data };
      for (const key of sensitiveKeys) {
        if (key in redacted.data) {
          redacted.data[key] = "[REDACTED]";
        }
      }
    }
    return redacted;
  }
}

class SessionImporter {
  loadFromJSON(jsonString) {
    const data = JSON.parse(jsonString);
    if (!data.session_id) {
      throw new AgentDeployError("Invalid session export: missing session_id");
    }
    return data;
  }

  replay(session, handler) {
    const results = { matched: 0, diverged: 0, divergences: [] };
    for (let i = 0; i < session.events.length; i++) {
      const event = session.events[i];
      if (!handler) {
        results.matched++;
        continue;
      }
      const actual = handler(event);
      if (JSON.stringify(actual) === JSON.stringify(event.data)) {
        results.matched++;
      } else {
        results.diverged++;
        results.divergences.push({ index: i, expected: event.data, actual });
      }
    }
    return results;
  }
}

module.exports = {
  VERSION,
  AgentDeployError,
  SessionExporter,
  SessionImporter,
};
