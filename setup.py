from setuptools import find_packages, setup

setup(
    name="topological-superfluid-ai",
    version="0.1.0",
    description="Topology-driven adaptive AI prototype with frozen prototypes and instant matching",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
    ],
)