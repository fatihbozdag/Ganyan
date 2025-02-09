from setuptools import setup, find_packages

setup(
    name="horse_racing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "pymc>=5.0.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.12.0",
        "jupyter>=1.0.0",
        "pytest>=7.3.0",
    ],
) 