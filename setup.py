from setuptools import setup, find_packages

setup(
    name = "AUTORCH",
    version = "0.1.0",
    description = ("help use pytorch eazy"),
    packages = find_packages(),
    url="",
    author = "ricky-yu",
    author_email = "skywalker0803r@gmail.com",
    install_requires = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "seaborn",
        "plotly",
        "torch",
        "tensorboard",
        "tqdm",
    ]
)