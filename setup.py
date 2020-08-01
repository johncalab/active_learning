from setuptools import setup, find_packages

dependencies = [
    "torch",
    "jupyterlab",
    "pytorch-nlp",
    "matplotlib",
    "sklearn",
]

setup(
    name="bald_stuff",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(include="bald"),
    install_requires=dependencies,
)
