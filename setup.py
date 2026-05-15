from setuptools import setup, find_packages

setup(
    name="prompt-probing",
    version="0.1.0",
    description=(
        "Zero-Shot Rankability: Revealing Latent Ordinal Structure "
        "in Multimodal Large Language Models via Language"
    ),
    author="KAIST AMI Lab",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
    ],
    extras_require={
        "hf": [
            "torch>=2.0",
            "torchvision>=0.15",
            "transformers>=4.37",
            "accelerate>=0.26",
        ],
        "openai": ["openai>=1.10"],
        "viz": ["matplotlib>=3.7", "seaborn>=0.12"],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
        ],
        "all": [
            "torch>=2.0",
            "torchvision>=0.15",
            "transformers>=4.37",
            "accelerate>=0.26",
            "openai>=1.10",
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "pytest>=7.4",
            "pytest-cov>=4.1",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
