from setuptools import setup, find_packages

setup(
    name = "autorch",
    version = "0.3.2",
    description = ("auto pytorch"),
    packages = find_packages(),
    url="https://github.com/skywalker0803r/autorch",
    author = "skywalker0803r",
    author_email = "skywalker0803r@gmail.com",
    install_requires = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "torch",
        "torch-dct",
        "tensorboard",
        "tqdm",
    ]
)
