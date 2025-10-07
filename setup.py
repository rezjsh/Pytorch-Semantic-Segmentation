from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Pytorch-Semantic-Segmentation"
AUTHOR_USER_NAME = "rezjsh"
SRC_REPO = "Semantic_Segmentation"
# AUTHOR_EMAIL = "your.email@example.com" # Replace with your actual email


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    # author_email=AUTHOR_EMAIL,
    description="A small python package for Semantic Segmentation with diffrent models containing  UNET Attention using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": SRC_REPO},
    packages=find_packages(where=SRC_REPO),
    install_requires=[
        "torch",
        "torchvision",
        "pandas",
        "Pillow",
        "transformers",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "PyYAML",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.7",
)