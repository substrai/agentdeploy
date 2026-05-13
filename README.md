# AgentDeploy

**Zero-to-production AI agent deployment framework.**

> Built by [SubstrAI](https://github.com/substrai) — Open-source GenAI frameworks for serverless infrastructure.

[![PyPI version](https://badge.fury.io/py/substrai-agentdeploy.svg)](https://pypi.org/project/substrai-agentdeploy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Building an AI agent is easy. Deploying it to production with auth, scaling, sessions, cost controls, and multi-tenancy takes weeks of custom infrastructure — every time.

## The Solution

```python
from agentdeploy import agent, Tool, Session

@Tool(description="Search knowledge base")
def search_kb(query: str) -> list:
    return ["result 1", "result 2"]

@agent(name="support-agent", model="bedrock/claude-3-sonnet", tools=[search_kb])
def support_agent(message: str, session: Session) -> str:
    return f"I can help with: {message}"
```

```bash
agentdeploy deploy --env prod
# ✓ Deployed: https://xxx.execute-api.us-east-1.amazonaws.com/prod/agent
```

## Features

- **@agent decorator** — turn any function into a deployable agent
- **Session management** — DynamoDB-backed conversation persistence
- **Tool sandboxing** — per-tenant tool permissions with audit trail
- **Cost circuit breakers** — auto-kill runs exceeding budget
- **Multi-tenancy** — tenant isolation, rate limiting, cost tracking
- **One-command deploy** — API Gateway + Lambda + DynamoDB
- **Provider agnostic** — works with Bedrock, OpenAI, Anthropic, custom
- **Adapter interface** — supports LangChain, CrewAI, Strands, custom agents

## Installation

```bash
pip install substrai-agentdeploy
```

## Quick Start

```bash
agentdeploy init my-agent
cd my-agent
agentdeploy dev    # local dev server
agentdeploy deploy --env prod
```

## License

MIT — see [LICENSE](LICENSE)

## Author

**Gaurav Kumar Sinha** — Founder, [SubstrAI](https://github.com/substrai)

- Email: gaurav@substrai.dev
- GitHub: [@substrai](https://github.com/substrai)
