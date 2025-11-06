.PHONY: build-docs docs preview clean help

help:
	@echo "Available targets:"
	@echo "  build-docs  - Build the Sphinx documentation"
	@echo "  docs        - Build and serve documentation with auto-reload"
	@echo "  preview     - Clean, build and preview documentation in browser"
	@echo "  clean       - Remove the documentation build directory"

clean:
	@echo "Cleaning documentation build directory..."
	rm -rf docs/_build

build-docs:
	@echo "Building documentation..."
	uv run sphinx-build -M html docs docs/_build

docs: build-docs
	@echo "Starting documentation server at http://localhost:8001"
	@echo "Press Ctrl+C to stop the server"
	cd docs/_build/html && uv run python -m http.server 8000

preview: clean build-docs
	@echo "Starting documentation server with auto-reload..."
	@echo "Documentation will be available at http://localhost:8000"
	uv run sphinx-autobuild docs docs/_build/html --port 8000 --open-browser
