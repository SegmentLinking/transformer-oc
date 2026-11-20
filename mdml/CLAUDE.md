# mdml/CLAUDE.md

Rules and guidelines for MDML.

## Rules

- Read-only commands are generally fine to run anytime, but please ask when in doubt.
- Do not run `git commit` or `git push`. Alex or a developer will handle that.
- Do not run `kubectl exec` or other commands which may write/modify, unless explicitly given consent by Alex or another developer to do that. Consent is valid for the subsequent response only. Consent is valid throughout the entire response, e.g., a command many be run multiple times to make that response.

## Guidelines

- If a piece of code is expected to take more than one minute to run, include an announcement that this piece of code is starting. For example, `print(f"[windows] path={path}")` indicates a new filepath will be opened to calculate windows, and this could be a computationally expensive process. Without an announcement, Alex might think the code is hanging.
