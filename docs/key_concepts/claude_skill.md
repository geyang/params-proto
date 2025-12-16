# Claude Skill

A Claude skill is a collection of documentation files that Claude Code and other Claude-based tools can use to provide
better assistance when working with a library or codebase.

params-proto includes a Claude skill that helps AI assistants work with the library effectively.

## Skill Structure

The skill provides:

1. **Quick Reference** - Cheat sheet for common patterns and syntax
2. **API Documentation** - Detailed docs for each decorator
3. **Feature Guides** - Help generation, environment variables, sweeps
4. **Examples** - Common patterns for ML training, multi-namespace configs

```
skill/
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

## Using the Skill

### Option 1: Add via Claude Code (Recommended)

Type in Claude Code:

```
# add https://raw.githubusercontent.com/geyang/params-proto/main/skill/index.md as a skill
```

### Option 2: Import in CLAUDE.md

Add this import to your project's `CLAUDE.md` file:

```markdown
@import https://raw.githubusercontent.com/geyang/params-proto/main/skill/index.md
```

### Option 3: Copy Locally

Copy the `skill/` folder to your project's `.claude/skills/params-proto/` directory.

## Contributing to the Skill

The skill files are located in `skill/` and are automatically generated from the documentation in `docs/`.
Feel free to contribute by filing an issue! 
