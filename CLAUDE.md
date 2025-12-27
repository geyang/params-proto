# Release Process

## Full Release (with PyPI publish)

### 1. Bump Version

Update `pyproject.toml`:
```toml
version = "3.0.0-rcX"
```

### 2. Commit and Push

```bash
git add -A && git commit -m "Bump version to 3.0.0-rcX"
git push origin main
```

### 3. Create Tags

```bash
git tag -f v3.0.0-rcX HEAD
git tag -f latest HEAD
git push origin refs/tags/v3.0.0-rcX --force
git push origin refs/tags/latest --force
```

### 4. Create GitHub Release

```bash
gh release create v3.0.0-rcX --title "v3.0.0-rcX" --latest --notes "## Changes
- Feature 1
- Feature 2"
```

### 5. Publish to PyPI

```bash
rm -rf dist && uv build
uv publish --token "$(pass show pypi | head -1)"
```

## Documentation-Only Update

When updating docs without a new PyPI release:

```bash
# Commit and push
git add -A && git commit -m "Update docs: description"
git push origin main

# Update latest tag for RTD
git tag -f latest HEAD
git push origin refs/tags/latest --force
```

Note: RTD uses the `latest` tag (not branch) to build documentation.
