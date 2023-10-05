#!/usr/bin/env sh

rm -rf _build

# Build the API documentation with autoapi
sphinx-apidoc -ePf --ext-autodoc --output-dir=_api -t _templates/ ../src/simfmri  "**/_version.py"

# Finally build the entire docs
jupyter-book build .
