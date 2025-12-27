# Release Process

## Full Release (with PR and PyPI publish)

### 1. Prepare Changes

Update `pyproject.toml` version:
```toml
version = "3.0.0-rcX"
```

Update `docs/release_notes.md` with changes.

Update `skill/quick-reference.md` version if needed.

### 2. Create PR

```bash
# Create feature branch
git checkout -b release/v3.0.0-rcX

# Commit all changes
git add -A && git commit -m "Release v3.0.0-rcX: brief description"

# Push and create PR
git push -u origin release/v3.0.0-rcX
gh pr create --title "Release v3.0.0-rcX" --body "## Changes
- Feature/fix 1
- Feature/fix 2"
```

### 3. Merge PR

```bash
gh pr merge --squash --delete-branch
git checkout main && git pull
```

### 4. Create Tags

```bash
git tag -f v3.0.0-rcX HEAD
git tag -f latest HEAD
git push origin refs/tags/v3.0.0-rcX --force
git push origin refs/tags/latest --force
```

### 5. Create GitHub Release

```bash
gh release create v3.0.0-rcX --title "v3.0.0-rcX" --latest --notes "## Changes
- Feature/fix 1
- Feature/fix 2"
```

### 6. Publish to PyPI

```bash
rm -rf dist && uv build
uv publish --token "$(pass show pypi | head -1)"
```

## Documentation-Only Update

When updating docs without a new PyPI release:

```bash
# Commit and push directly to main
git add -A && git commit -m "Update docs: description"
git push origin main

# Update latest tag for RTD
git tag -f latest HEAD
git push origin refs/tags/latest --force
```

## Quick Fixes (No PR needed)

For small fixes that don't need review:

```bash
# Commit and push to main
git add -A && git commit -m "Fix: description"
git push origin main

# If releasing: follow steps 4-6 above
```

## Notes

- RTD uses the `latest` tag (not branch) to build documentation
- Always update `docs/release_notes.md` for code changes
- Update `skill/quick-reference.md` version number for new releases
