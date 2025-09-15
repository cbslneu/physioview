import setuptools

# Dependencies
requirements = [
    'cvxopt',
    'cython==0.29.21',
    'dash==2.17.1',
    'dash_bootstrap_components==1.6.0',
    'dash_daq==0.5.0',
    'dash_uploader',
    'diskcache',
    'flirt',
    'matplotlib',
    'multiprocess',
    'numpy==1.26.4',
    'openpyxl',
    'packaging==21.3',
    'pandas~=2.3.0',
    'plotly==6.2.0',
    'psutil',
    'pyEDFlib',
    'scipy',
    'tqdm==4.65.0'
]

setuptools.setup(
    name = 'physioview',
    version = '1.0',
    author = 'Natasha Yamane, Varun Mishra, and Matthew S. Goodwin',
    description = 'A signal quality assessment pipeline for wearable '
                  'physiological data',
    license = 'GPL-3.0',
    packages = setuptools.find_packages(),
    install_requires = requirements,
    url = 'https://github.com/cbslneu/heartview',
    python_requires = '>=3.9.23'
)