<div align="center" style="margin-bottom: 1rem;">
  <h1 style="font-size: 2rem;margin: 0;"> reviewbot </h1>

  <img src="https://img.shields.io/badge/python-3.13-3776AB.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/langchain-1.2%2B-blue.svg" alt="LangChain" />
  <img src="https://img.shields.io/badge/fastapi-webhook-green.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/gitlab-mr%20reviews-orange.svg" alt="GitLab" />
</div>

Multi-agent LangChain reviewer for GitLab MRs. Pulls diffs, reasons across codebases, and
posts actionable review notes back to the MR.

![ReviewBot](https://i.imgur.com/cPl9isr.png)

## What is Reviewbot?

Reviewbot wires together diff fetching, codebase context, and LLM reasoning to automate code
reviews on GitLab:

- **Multi-Agent Review Flow**. Coordinates tasks like diff inspection, context lookup, and issue
  synthesis
- **MR-Centric**. Works on GitLab MRs and posts discussions/notes back to the MR
- **Codebase Context**. Optional embeddings + search for better review depth
- **Ignore Rules**. Supports `.reviewignore` and global ignore patterns to skip noise
- **CLI + Webhook**. Run manually or trigger from GitLab pipeline/note events
