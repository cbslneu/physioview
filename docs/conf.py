# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PhysioView'
copyright = '2025, Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
author = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
release = '1.0'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '../assets/physioview-logo.png'
html_favicon = '../assets/favicon.ico'
html_css_files = [
    'css/custom-furo.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#ee8a78",
        "color-brand-content": "#ee8a78",
        "color-link--visited": "#ee8a78",
        "color-brand-visited": "#ee8a78",
        "color-admonition-background": "#f0f0f0",
        "color-admonition-title-background--seealso": "#f0f0f0",
        "color-admonition-title--seealso": "#ee8a78",
        "color-sidebar-background": "#47555e",
        "color-sidebar-search-background": "#47555e",
        "color-sidebar-search-icon": "#ffffff",
        "color-sidebar-search-background--focus": "#47555e",
        "color-sidebar-search-foreground": "#ffffff",
        "color-sidebar-search-border": "#47555e",
        "color-sidebar-link-text--top-level": "#ffbbae",
        "color-api-name": "#de7765",
        "color-api-pre-name": "#de7765",
        "color-sidebar-caption-text": "#ffffff",
        "color-content-foreground": "#444444",
        "color-background-hover": "none",
        "color-foreground-secondary": "#444444",
        "code-font-size": "10pt",
        "toc-item-spacing-horizontal": "0.4rem"
    },
    "dark_css_variables": {
        "color-brand-primary": "#ffbbae",
        "color-brand-content": "#ffbbae",
        "color-link--visited": "#ffbbae",
        "color-brand-visited": "#ffbbae",
        "color-admonition-background": "#444d54",
        "color-admonition-title-background--seealso": "#444d54",
        "color-admonition-title--seealso": "#ee8a78",
        "color-background-primary": "#313c42",
        "color-code-background": "#313c42",
        "color-sidebar-background": "#313c42",
        "color-sidebar-search-background": "#313c42",
        "color-sidebar-search-icon": "#ffffff",
        "color-sidebar-search-background--focus": "#313c42",
        "color-sidebar-search-foreground": "#ffffff",
        "color-sidebar-search-border": "#313c42",
        "color-api-name": "#ffc5be",
        "color-api-pre-name": "#ffc5be",
        "color-sidebar-caption-text": "#ffffff",
        "color-content-foreground": "#ffffff",
        "color-background-hover": "none",
        "color-background-secondary": "#262e33",
        "color-highlight-on-target": "#262e33",
        "color-foreground-secondary": "#ffffff",
        "code-font-size": "10pt",
        "toc-item-spacing-horizontal": "0.4rem"
    },
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/cbslneu/physioview/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/cbslneu/physioview",
                "html": """
                    <i class="fa-brands fa-github fa-lg"></i>
                """,
                "class": "",
            },
        ],
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_light_style = 'friendly'
pygments_dark_style = 'lightbulb'

# -- Options for LaTeX output ------------------------------------------------
authors = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin'
toctree_only = True
latex_documents = [
    ('index', 'physioview.tex', 'PhysioView Documentation',
     authors.replace(', ', '\\and ').replace(' and ', '\\and and '),
     'manual', toctree_only)
]