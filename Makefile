.PHONY: docs preview

# Build documentation
build-docs:
	rm -rf docs/_build
	uv run --extra docs sphinx-build docs docs/_build/html

# Build and serve documentation
docs: build-docs
	cd docs/_build/html && python -m http.server 8000

# Live-reload documentation server
preview: build-docs
	uv run --extra docs sphinx-autobuild --host 0.0.0.0 docs docs/_build/html
