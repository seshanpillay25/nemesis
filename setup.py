#!/usr/bin/env python3
"""
Nemesis: Your model's greatest adversary and teacher.

Setup configuration for the nemesis-ai package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nemesis-ai",
    version="0.1.0",
    author="Nemesis Contributors",
    description="Your model's greatest adversary - adversarial robustness toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seshanpillay25/nemesis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "tensorflow": [
            "tensorflow>=2.8.0; python_version<'3.12'",
            "tensorflow>=2.15.0; python_version>='3.12'",
        ],
    },
    entry_points={
        "console_scripts": [
            "nemesis-battle=nemesis.cli:main",
        ],
    },
    keywords="adversarial machine-learning robustness security ai",
    project_urls={
        "Source": "https://github.com/seshanpillay25/nemesis",
    },
    include_package_data=True,
    zip_safe=False,
)