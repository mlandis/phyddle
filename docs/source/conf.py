# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../phyddle/src'))


# -- Project information

project = 'phyddle'
copyright = '2025'
author = 'Michael Landis, Ammon Thompson'

release = '0.3.0'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
    'sphinx_subfigure',
    'sphinxemoji.sphinxemoji',
]

if os.getenv("GITHUB_ACTIONS"):
    extensions.append("sphinxcontrib.googleanalytics")
    googleanalytics_id = "G-KW8RTLPKH4"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_logo  = 'images/phyddle_logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_baseurl = 'phyddle.org'
html_extra_path = ['CNAME']

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Subfigure settings
numfig = True
