.PHONY: docs preview

# Build and serve documentation
docs:
	rm -rf docs/_build
	uv run --extra docs sphinx-build docs docs/_build/html
	cd docs/_build/html && python -m http.server 8888

# Live-reload documentation server
preview:
	uv run --extra docs sphinx-autobuild --host 0.0.0.0 docs docs/_build/html
