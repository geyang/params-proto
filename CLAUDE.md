# Release Process

## Version Tags

```bash
git tag v3.0.0-rcX
git push origin v3.0.0-rcX
```

## Read the Docs (latest tag)

RTD uses a `latest` tag (not branch) to build documentation.

```bash
git tag -f latest HEAD
git push origin refs/tags/latest --force
```
