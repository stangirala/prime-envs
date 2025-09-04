# hello-world-multi-turn

### Overview
- **Environment ID**: `hello-world-multi-turn`
- **Short description**: Standalone example to demonstrate how multi-turn examples work.
- **Tags**: hello-world, multi-turn

### Datasets
- **Primary dataset(s)**: N/A
- **Source links**: https://github.com/stangirala/prime-envs
- **Split sizes**: N/A

### Task
- **Type**: multi-turn
- **Parser**: XMLParser
- **Rubric overview**: Dummy reward, return 1.0 to keep the conversation moving.

### Quickstart
Run an evaluation with default settings:

Setup OAI keys with `OPENAI_API_KEY` environment variable.

For the dummy dataset with one record, use the follow to test the code,

```bash
uv run vf-eval hello-world-multi-turn -m gpt-4.1-mini -n 1 --save-dataset

```

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Dummy reward |

