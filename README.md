<div align="center" style="margin-bottom: 1rem;">
  <h1 style="font-size: 2rem;margin: 0;"> reviewbot </h1>

  <a href="https://github.com/canefe/reviewbot/releases"><img src="https://img.shields.io/github/v/release/canefe/reviewbot?include_prereleases&style=flat-square" alt="Latest Version" /></a>&nbsp;
  <a href="https://github.com/canefe/reviewbot/actions"><img src="https://img.shields.io/github/actions/workflow/status/canefe/reviewbot/ci.yml?style=flat-square" alt="Build Status" /></a>

  <p>
    <img src="https://img.shields.io/badge/python-3.13-3776AB.svg?style=flat-square" alt="Python Version" />
    <img src="https://img.shields.io/badge/langchain-1.2%2B-blue.svg?style=flat-square" alt="LangChain" />
  </p>
</div>

Multi-agent LangChain reviewer for GitLab MRs. Pulls diffs, reasons across codebases, and
posts actionable review notes back to the MR.

![ReviewBot](https://i.imgur.com/YDxYiDE.png)

## What is Reviewbot?

Reviewbot wires together diff fetching, codebase context, and LLM reasoning to automate code
reviews on GitLab:

- **Multi-Agent Review Flow**. Coordinates tasks like diff inspection, context lookup, issue
  synthesis, and issue validation to reduce hallucinations
- **MR-Centric**. Works on GitLab MRs and posts discussions/notes back to the MR
- **Codebase Context**. Optional embeddings + search for better review depth
- **Ignore Rules**. Supports `.reviewignore` and global ignore patterns to skip noise
- **CLI + Webhook**. Run manually or trigger from GitLab pipeline/note events


