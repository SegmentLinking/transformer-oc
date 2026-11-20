# mdml/CLAUDE.md

Rules and guidelines for MDML.

## Rules

- Read-only commands are generally fine to run anytime, but please ask when in doubt.
- Do not run `git commit` or `git push`. Alex or a developer will handle that.
- Do not run `kubectl exec` or other commands which may write/modify, unless explicitly given consent by Alex or another developer to do that. Consent is valid for the subsequent response only. Consent is valid throughout the entire response, e.g., a command many be run multiple times to make that response.

## Guidelines

- Each step of a script’s `main` that loops over events or could take more than a few seconds prints a `[tag] key=value` line when it starts. Print the current time if there is not a meaningful `key=value` to print. Speed differs between machines (a step that takes seconds on phi3 took 15 minutes on NRP), so a silent step looks like a hang.

