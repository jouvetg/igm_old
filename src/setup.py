#!/usr/bin/env python
# Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3)

from setuptools import setup

setup(
    name="igm",
    version="1.0.0",
    author="Guillaume Jouvet",
    author_email="guillaume.jouvet@unil.ch",
    description="The Instructed Glacier Model",
    url="https://github.com/jouvetg/igm",
    license="gpl-3.0",
    py_modules=["igm"],
    scripts=["igm.py"],
)
