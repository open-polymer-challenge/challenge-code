from setuptools import setup, find_packages
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "opc"))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="opc",
    version="0.1.0",
    description="Open Polymer Challenge",
    url="https://github.com/open-polymer-challenge/challenge-code",
    author="open-polymer-challenge",
    author_email="openpolymerchallenge@gmail.com",
    keywords=[
        "AI for Science",
        "Polymer Informatics",
        "Polymer Property Prediction",
        "Inverse Polymer Design",
        "Graph Machine Learning",
        "Pytorch",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=["examples"]),
)
