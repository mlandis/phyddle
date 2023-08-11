#!/usr/bin/env sh

# rebuild local phyddle install
pip install -e .

# rebuild sphinx pages
cd docs
make html
cd ..

# get git info
GIT_BRANCH=$(cat .git/HEAD | cut -f3 -d'/')
GIT_COMMIT=$(cat .git/refs/heads/${GIT_BRANCH} | cut -c 1-7)
echo "${GIT_BRANCH} ${GIT_COMMIT}" > git_info.txt
