#!/usr/bin/python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
deps = ['matplotlib', 'multiprocess', 'scipy', 'numpy', 'matplotlib', 'pandas', 'pathos', 'PyYAML', 'Shapely']

setuptools.setup(
    name="neutpy",
    version=open("neutpy/_version.py").readlines()[-1].split()[-1].strip("\"'"),
    author="Maxwell D. Hill, Jonathan J. Roveto",
    install_requires=deps,
    author_email="max.hill@pm.me, veto1024@gmail.com",
    description="NeutPy - A neutrals code for tokamak fusion reactors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gt-frc/neutpy/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.8',
)

if __name__ == '__main__':
    print("Test")
    pass
