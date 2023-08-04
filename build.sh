#!/usr/bin/env sh

# rebuild local phyddle install
pip install -e .

# rebuild sphinx pages
cd docs
make html
cd ..
