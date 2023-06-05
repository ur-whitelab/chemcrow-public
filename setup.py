import os
from glob import glob
from setuptools import setup, find_packages

exec(open("chemcrow/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chemcrow",
    version=__version__,
    description="Collection of chemistry tools for use with language models",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/chemcrow-public",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "synspace",
        "molbloom",
        "paper-qa==1.1.1",
        "rxn4chemistry",
        "google-search-results",
        "pandas",
        "langchain==0.0.173",
        "nest_asyncio",
        "ipywidgets",
        "ipykernel",
        "tiktoken",
        "rmrkl @ git+https://github.com/doncamilom/robust-mrkl.git",
        "python-dotenv",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
