
This guide explains how to update and maintain phyddle, as a
a publicly available software package.

Main technologies:
    - GitHub as version control software
    - Sphinx for documentation
    - PyPI and TestPyPI to release package
    - bump2version to manage version-strings
    - Conda to provide virtual environments

-----

# Standard update procedure

These are the steps to release a new version of phyddle (main branch). It is
not possible to "undo" a new version. Bad versions must be patched and assigned
a new version number. Go slowly, read the output from each step, and verify
everything looks okay as you go.

```
# enter phyddle project directory
cd ~/projects/phyddle

# enter main branch
git checkout main

# merge in development changes
git merge development

# add notes to docs
vim docs/source/updates.rst

# prepare source code for update
git commit -am 'preparing for version update'

# bump versions
bump2version patch                                
git commit -am 'apply version update to x.x.x'

# build pypi
python3 -m build

# upload dist to testpypi
python3 -m twine upload --skip-existing --repository testpypi dist/phyddle-x.x.x.*

# upload dist to pypi
twine upload dist/phyddle-x.x.x.*

# build conda project
mamba build . -c bioconda

# authenticate with anaconda
anaconda login

# upload to anaconda
anaconda upload \
    /Users/mlandis/opt/miniconda3/conda-bld/osx-64/phyddle-x.x.x-py39_0.conda

```


-----

# GitHub repository

We use `git` and GitHub as version control software. We have three main branches
- `main` : supported and tested features, stable
- `development` : experimental and untested features, unstable
- `gh-pages` : used to display Sphinx documentation

In general, `main` should only be updated by merging when all code in `development`
is tested and works. No other commits should enter the `main` history.

The GitHub Pages branch `gh-pages` is built automatically and should not be modified.

Most new features should be developed in new branches derived from development, e.g.

```
# switch to development
git checkout development
# get any recent commits
git pull
# make new branch and switch
git checkout -b my_new_feature
# edit source
vim src/phyddle/utilities
# commit changes to new-branch
git commit -am 'simply amazing'
# push new branch to origin (GitHub)
git push --set-upstream my_new_feature
# verify test
pytest tests
# visit GitHub and verify new-branch passes tests
cowsay good luck
# switch back to development
git checkout development
# merge new branch into development
git merge my_new_feature
# verify tests pass locally
pytest tests
# push development to origin (GitHub)
git push
# visit GitHub and verify development+new-branch passes tests
cowsay ...and again
# done
```

See documentation for how to push the packae from `main`.


-----

# Documentation with Sphinx

Sphinx generates static HTML content based on the current version
of phyddle. This means you need to rebuild phyddle if you want
new code/comments/docstrings to appear in the Sphinx HTML.

Source files in reStructuredText (rst) format are here:

    ./docs/source

Local build:
    
    cd docs
    make html
    open build/html/index.html

Notes:
- Needed to add `sphinx.ext.napoleon` to `extensions` in
  `docs/source/conf.py` to support Google docstrings.
- Needed to run `python3 -m pip install rtd-theme` for ReadTheDocs theme.


-----

# GitHub Pages for Sphinx Docs


phyddle documentation is published online here:

    https://mlandis.github.io/phyddle/

We use GitHub pages to host Sphinx documentation, following the
strategy outlined here:

    https://coderefinery.github.io/documentation/gh_workflow/

GitHub Actions automatically rebuilds Sphinx with every push, pull_request,
and workflow_dispatch event for the `main` branch of phyddle. Config
file is hosted here:

    ./.github/workflows/documentation.yml

Setting for GitHub Pages are managed here

    https://github.com/mlandis/phyddle/settings/pages


-----

# Software version with bump2version

bump2version is a Python tool that will search-replace any
version-string a set of files.

Version string is MAJOR.MINOR.PATCH

Config file for bump2version:

    ./.bumpversion.cfg

To bump the patch version:

    bump2version patch


# Package build with PyPI and TestPyPI

PyPI is a repository that pip uses to install published Python
packages. TestPyPI is a staging area for the public release.


PyPI public package for phyddle is here:

    https://pypi.org/project/phyddle/

TestPyPI for testing phyddle package deployment is here:

    https://test.pypi.org/project/phyddle/

## Deployment

We followed these instructions to upload phyddle with PyPI
and TestPyPI

    https://packaging.python.org/en/latest/tutorials/packaging-projects/

Project settings are stored here:

    ./pyproject.toml


Create built package in `dist`

    # install/upgrade build package
    python3 -m pip install --upgrade build

    # uses pyproject.toml
    python3 -m build          

Upload to distribution to TestPyPI
  
    # install/upgrade twine package
    python3 -m pip install --upgrade twine

    # upload dist to testpypi with twine
    python3 -m twine upload --repository --skip-existing testpypi dist/*

    # verify TestPyPI install works (e.g. for x.x.x => v0.0.7)
    python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps phyddle-x.x.x.*

    # should be able to import within Python session now, e.g.
    # >>> from phyddle import command_line


Upload distribution to PyPI (similar to above)
    
    # install/upgrade twine package
    python3 -m pip install --upgrade twine

    # upload dist to pypi with twine
    twine upload dist/*

    # verify PyPI install works
    python3 -m pip install phyddle

    # should be able to import within Python session now, e.g.
    # >>> from phyddle import command_line


-----

# Command line tool, `phyddle`

The phyddle command line tool (`phyddle`) is implemented in this directory:

    ./src/phyddle/command_line:run()

The `phyddle` command is packaged through `pyproject.toml`

    [project.scripts]
    phyddle = "phyddle.command_line:run"


-----

-----

# Automatic testing with PyTest

PyTest will execute all test scripts (.py) and create errors if any test
functions with the `test_` prefix create errors when executed.

phyddle tests are stored here:

    ./tests/

GitHub Actions automatically runs all PyTest  Sphinx with every push, pull_request,
and workflow_dispatch event for the GitHub repository for phyddle. Config
file is hosted here:

    .github/workflows/tests.yml

---------

# conda

### Installing conda tools

Install anaconda from website?
Optional: install blackmamba/boa to speed up install (don't remember how).
```
brew install miniconda
brew install conda
conda update -n base -c conda-forge conda
```

I used mambaforge (C++ implementation of conda [faster])
```
brew install mambaforge

Ensure `anaconda` and `conda` are in PATH

### Installing from conda

New install on new system should be easy. 

```
conda create -n phyddle
conda activate phyddle
conda install -c bioconda -c landismj phyddle
```

### Updating conda package

Updating conda with new phyddle version is also not too hard. You must register
an account with anaconda.org. 

Build sequence might take 15 minutes.

```
cd ~/projects/phyddle
# need -c bioconda for dendropy
conda build . -c bioconda
# (optional) using mambabuild, faster
conda mambabuild . -c bioconda
```

Create anaconda account and login
```
anaconda login
# enter user/password
anaconda upload \
    /Users/mlandis/opt/miniconda3/conda-bld/osx-64/phyddle-x.x.x-py39_0.conda
```


File is uploaded here: https://anaconda.org/landismj/phyddle



### Setting up conda


Useful links:
https://docs.conda.io/projects/conda-build/en/stable/user-guide/tutorials/build-pkgs.html
https://docs.anaconda.com/free/anacondaorg/user-guide/tasks/work-with-packages/

- `meta.yml` defines the conda build design
- `build.sh` -- Unix/Mac install shell script
- `bld.bat` -- Windows install batch file


I could not get `conda build` to work the standard approach that relies on `meta.yml`, `build.sh`, `bld.bat`,  `setup.py`i and `source: git_url` or `source: git_rev`.

Instead, we use the PyPI `.tar.gz` as the source and build using `pyproject.toml`.

```
url: <get from PyPI Manage Project>
sha: <get from PyPI Manage Project> 
```

This was pretty annoying to setup.


