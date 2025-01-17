from setuptools import setup, find_packages

setup(
    name="dynasor",
    version="0.1.0",
    description="Dynamic Reasoning Framework for LLMs",
    author="Dynasor Team",
    packages=find_packages(),
    install_requires=[
        "vllm",
        "tqdm",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
