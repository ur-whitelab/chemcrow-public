import os
from glob import glob
from setuptools import setup, find_packages

exec(open("chemcrow/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chemcrow",
    version=__version__,
    description="Accurate solution of reasoning-intensive chemical tasks, poweredby LLMs.",
    author="Andres M Bran, Sam Cox, Andrew White, Philippe Schwaller",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/chemcrow-public",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "synspace",
        "molbloom",
        "paper-qa==1.1.1",
        "google-search-results",
        "pandas",
        "langchain==0.0.173",
        "nest_asyncio",
        "ipywidgets",
        "ipykernel",
        "tiktoken",
        "rmrkl",
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
