import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynasor",
    version="0.0.1",
    author="Dynasor Team",
    description="Dynasor is a Python library for speedup LLM reasoning",
    packages=setuptools.find_packages(),
    install_requires=[
        "openai",
        "rich",
        "prompt_toolkit",
        "pydantic",
        "datasets",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)