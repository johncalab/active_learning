from setuptools import setup, find_packages

dependencies = [
    "torch",
    "pytorch-nlp",
    "sklearn",
    "datasets",
    "mlflow",
]

setup(
    name="bald_stuff",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(),
    install_requires=dependencies,
)
