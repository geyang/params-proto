# Claude Skill

params-proto includes a Claude skill that helps AI assistants work with the library effectively.

## What is a Claude Skill?

A Claude skill is a collection of documentation files that Claude Code and other Claude-based tools can use to provide better assistance when working with a library or codebase.

## Using the Skill

### In Claude Code

If you're using Claude Code in a project that depends on params-proto, point to the skill directory:

```bash
# In your project's .claude/settings.json or CLAUDE.md
# Reference the skill from the installed package or cloned repo
```

### Importing the Skill

You can reference the skill files from the params-proto repository:

```
.claude/skills/params-proto/
├── index.md              # Overview and navigation
├── quick-reference.md    # Cheat sheet
├── api/
│   ├── proto-cli.md      # @proto.cli decorator
│   ├── proto-prefix.md   # @proto.prefix decorator
│   └── types.md          # Type annotations
├── features/
│   ├── help-generation.md    # Auto help text
│   ├── environment-vars.md   # EnvVar support
│   └── sweeps.md             # Hyperparameter sweeps
└── examples/
    └── patterns.md       # Common patterns
```

### Via Raw GitHub URL

You can reference the skill files directly from GitHub:

```
https://raw.githubusercontent.com/geyang/params-proto/main/.claude/skills/params-proto/index.md
```

## Skill Contents

The skill provides:

1. **Quick Reference** - Cheat sheet for common patterns and syntax
2. **API Documentation** - Detailed docs for each decorator
3. **Feature Guides** - Help generation, environment variables, sweeps
4. **Examples** - Common patterns for ML training, multi-namespace configs

## Contributing to the Skill

The skill files are located in `.claude/skills/params-proto/`. PRs welcome to improve the documentation!
