from setuptools import setup, find_packages

setup(
    name="obscodec",
    version="2.0.0-alpha",
    description="Semantic communication codec benchmark for multi-agent MPE observations — Phase 2a",
    author="MacswareX",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "pettingzoo>=1.24",
        "gymnasium>=0.29",
        "matplotlib>=3.7",
    ],
)
