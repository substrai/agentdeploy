# AgentDeploy AWS Benchmark Results

**Date:** May 13, 2026
**Region:** us-east-1
**Model:** anthropic.claude-3-haiku-20240307-v1:0
**Runtime:** Python 3.14, macOS (Apple Silicon)

---

## Summary

| Metric | Value |
|--------|-------|
| **Framework overhead** | 0.006 ms per invocation |
| **Overhead as % of LLM call** | 0.000% (negligible) |
| **Session create+save** | 0.002 ms |
| **Bedrock avg latency** | 1,572 ms |
| **Tool sandbox** | Correctly blocks denied tools |
| **Cost enforcement** | Blocks over-budget sessions |
| **Rate limiting** | Correctly blocks at limit |
| **Canary routing** | ~80/20 split verified |
| **Observability** | Full 10-event trace captured |

---

## Benchmark 1: Agent Runtime Overhead

100 agent invocations (local mode):

| Metric | Value |
|--------|-------|
| Mean | 0.006 ms |
| Median | 0.005 ms |
| P95 | 0.007 ms |
| P99 | 0.019 ms |

**Conclusion:** AgentDeploy adds 6 microseconds of overhead — completely invisible.

---

## Benchmark 2: Real Agent with Bedrock

3 customer support questions answered by Claude 3 Haiku:

| Question | Latency | Tokens |
|----------|---------|--------|
| What is your return policy? | 1,689 ms | 31 in / 129 out |
| How do I track my order? | 1,440 ms | 32 in / 79 out |
| Can I change my shipping address? | 1,588 ms | 32 in / 101 out |

**Average latency: 1,572 ms**
**Framework overhead: 0.000% of total**

---

## Benchmark 3: Session Management

| Metric | Value |
|--------|-------|
| Create + save (100 iterations) | 0.002 ms mean |
| P99 | 0.005 ms |
| Multi-turn (20 turns) | 40 messages stored |
| Active sessions | 101 |

---

## Benchmark 4: Tool Sandbox

| Scenario | Result |
|----------|--------|
| Enterprise + search_kb | ALLOWED |
| Free-tier + create_ticket | BLOCKED (denied) |
| LLM tool decision | Correctly chose "create_ticket" |

---

## Benchmark 5: Cost Enforcement

| Metric | Value |
|--------|-------|
| Session cost (10 requests) | $0.0500 |
| Daily remaining | $0.9500 |
| Over-budget blocked | Yes (action: block) |
| Actual Bedrock cost (3 calls) | $0.000410 |

---

## Benchmark 6: Multi-Tenancy

| Tenant | Rate Limit | Budget | Status |
|--------|-----------|--------|--------|
| Enterprise | 500/min | $200/day | ALLOWED |
| Free-tier | 10/min | $5/day | BLOCKED (after 12 requests) |

Tenants have fully isolated usage tracking.

---

## Benchmark 7: Versioning & Canary

| Step | Result |
|------|--------|
| Deploy v1.0 | Active (100% traffic) |
| Deploy canary v1.1 (20%) | Traffic split: 78/22 (close to 80/20) |
| Promote canary | v1.1 becomes active |
| Rollback | v1.0 restored |

---

## Benchmark 8: Observability

| Metric | Value |
|--------|-------|
| Trace events | 10 (full lifecycle) |
| Tool calls tracked | 1 |
| Tokens tracked | 100 |
| Cost tracked | $0.000092 |
| Duration | 1,228 ms |

---

## How to Reproduce

```bash
pip install substrai-agentdeploy[aws]
aws configure
python benchmarks/run_aws_benchmark.py
```

## Cost

Running the full benchmark costs approximately $0.001 (less than 1 cent) — 4 Bedrock Haiku calls.
