def review_prompt() -> str:
    return """
Review this GitLab merge request.

Hard rules:
- You are given tools to help you understand the codebase. You must use the tools to get greater context about the codebase. The code is already passed the compile check, do not assume there might be compiling issues.
- Use read_file tool to read the file at the given path, which can be retrieved from the search_codebase tool.
- Use the tool to get greater context about the codebase. You must understand the codebase before giving any valuable feedback.
- Every “issue” MUST include:
  - file path
  - a short quoted code fragment from the diff that proves it
  If you can’t quote evidence from the diff, DO NOT mention it.
- The code is already compiled, linted, and formatted, do not mention these issues.
- No tables.
Output format:
- Summary (1-3 bullets, diff-backed only)
- High-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Medium-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Low-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Suggestions (bullets; include snippets when helpful)

## High-risk issues (bullets, include file paths)
### Title_1
- Description_1
### Title_2
- Description_2
### Title_3
- Description_3

## Suggestions (bullets, include code snippets when helpful)
### Title_1
```diff
- var test := "test"
+ var test := "thats going to be a problem"
```
### Title_2
- Description_2
### Title_3
- Description_3

    """
