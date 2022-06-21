#!/usr/bin/env python
# Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@geo.uzh.ch>
# Published under the GNU GPL (Version 3)

from setuptools import setup

setup(
    name="igm",
    version="1.0.0",
    author="Guillaume Jouvet",
    author_email="guillaume.jouvet@geo.uzh.ch",
    description="The Instructed Glacier Model",
    url="https://github.com/jouvetg/igm",
    license="gpl-3.0",
    install_requires=[
        "protobuf<=4.0.0",
        "tensorflow",
        "tensorflow-addons",
        "scipy",
        "matplotlib",
        "xarray",
        "netCDF4",
        "IPython",
        "keras",
    ],
    extras_require={"gpu": ["tensorflow<=2.4.0", "tensorflow-gpu<=2.4.0"]},
    py_modules=["igm"],
    scripts=["igm.py"],
)
