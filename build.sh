#!/bin/sh

# rebuild local phyddle install
pip install .
# rebuild sphinx pages
cd docs
make html
cd ..
