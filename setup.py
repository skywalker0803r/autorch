from setuptools import setup, find_packages

setup(
    name = "AUTORCH",
    version = "0.1.1",
    description = ("help use pytorch eazy"),
    packages = find_packages(),
    url="https://github.com/skywalker0803r/autorch",
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
