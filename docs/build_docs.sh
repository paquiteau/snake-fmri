#!/usr/bin/env sh

rm -rf _build _api

# Build the API documentation with autoapi
sphinx-apidoc -ePf --ext-autodoc --output-dir=_api -t _templates/ ../src/snkf  "**/_version.py"

# Finally build the entire docs
jupyter-book build .
