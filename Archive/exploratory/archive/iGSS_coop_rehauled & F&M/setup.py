"""
Setup script for IGSS Framework.

Installation:
    pip install -e .

Dependencies:
    - numpy
    - matplotlib
    - networkx
    - mesa
    - sympy
    - deap
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="igss-framework",
    version="1.0.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="Integrated Genetic Programming for Social Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[Your Repository URL]",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
        "mesa>=1.0.0",
        "sympy>=1.8",
        "deap>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "igss=igss_framework.main:main",
        ],
    },
)
