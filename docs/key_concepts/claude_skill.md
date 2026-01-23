# Claude Skill

A Claude skill is a collection of documentation files that Claude Code and other Claude-based tools can use to provide
better assistance when working with a library or codebase.

params-proto includes a Claude skill that helps AI assistants work with the library effectively.

## Skill Structure

The skill follows the standard Claude skill format:

```
skill/
├── SKILL.md              # Main entry point with frontmatter
└── references/           # Detailed documentation
    ├── cli-and-types.md  # @proto.cli, @proto.prefix, type system
    ├── environment-vars.md   # EnvVar with templates and inheritance
    ├── sweeps.md         # piter and Sweep for hyperparameter search
    └── patterns.md       # Common ML patterns and examples
```

### SKILL.md

The main skill file contains:
- YAML frontmatter with `name` and `description` (triggers skill loading)
- Quick start guide and essential patterns
- Links to detailed reference files

### Reference Files

Detailed documentation for specific features:
- **cli-and-types.md** - Decorators, CLI parsing, type annotations, help generation
- **environment-vars.md** - EnvVar syntax, OR operations, type conversion, templates
- **sweeps.md** - piter and Sweep for systematic hyperparameter exploration
- **patterns.md** - Real-world ML config patterns and best practices

## Using the Skill

### Option 1: Add via Claude Code (Recommended)

Type in Claude Code:

```
/add-skill https://raw.githubusercontent.com/geyang/params-proto/main/skill/SKILL.md
```

### Option 2: Import in CLAUDE.md

Add this import to your project's `CLAUDE.md` file:

```markdown
@import https://raw.githubusercontent.com/geyang/params-proto/main/skill/SKILL.md
```

### Option 3: Copy Locally

Copy the `skill/` folder to your project's `.claude/skills/params-proto/` directory.

## What the Skill Provides

When active, the skill helps Claude:

1. **Create CLI applications** - Generate properly typed CLI entry points with auto-generated help
2. **Configure ML experiments** - Set up multi-namespace configurations for training scripts
3. **Use environment variables** - Read config from env vars with type conversion
4. **Run hyperparameter sweeps** - Use piter or Sweep for systematic exploration
5. **Handle Union types** - Create subcommand-like CLI patterns

## Contributing to the Skill

The skill files are located in `skill/` directory. To contribute:

1. Edit files in `skill/` or `skill/references/`
2. Keep SKILL.md concise (<500 lines)
3. Move detailed content to reference files
4. Test with Claude Code to verify behavior

Feel free to file an issue on GitHub with suggestions!
