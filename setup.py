from setuptools import setup, find_packages

setup(
    name="quant_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "yfinance>=0.2.40",
    ],
    python_requires=">=3.10",
)
