from setuptools import setup, find_packages

package_name = "kg-bilm"
version = {}
with open(f"{package_name}/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="kg-bilm",
    version=version["__version__"],
    author="czr",
    author_email=f"zrchen@tju.edu.cn",
    url=f"https://github.com/zirui-chen/{package_name}",
    description=f"The official {package_name} library",
    python_requires=">=3.11",
    packages=find_packages(include=[f"{package_name}*"]),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "peft",
        "transformers>=4.43.1,<=4.44.2",
        "datasets",
        "evaluate",
        "scikit-learn",
    ],
    extras_require={
        "evaluation": ["mteb>=1.14.12"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
