# AgentDeploy Benchmarks

Real AWS Bedrock benchmarks demonstrating AgentDeploy performance.

## Prerequisites

```bash
pip install substrai-agentdeploy[aws]
aws configure
```

## Running

```bash
python benchmarks/run_aws_benchmark.py
```

## What It Tests

| # | Benchmark | Description |
|---|-----------|-------------|
| 1 | Runtime Overhead | Framework overhead per invocation |
| 2 | Real Bedrock | End-to-end agent with Claude 3 Haiku |
| 3 | Sessions | Create/save/multi-turn performance |
| 4 | Tool Sandbox | Permission enforcement on real LLM decisions |
| 5 | Cost Enforcement | Budget blocking and tracking |
| 6 | Multi-Tenancy | Isolated rate limiting and usage |
| 7 | Versioning | Canary routing and rollback |
| 8 | Observability | Full request trace capture |

## Cost

< $0.001 per run (4 Bedrock Haiku calls).
