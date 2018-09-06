from setuptools import setup
import os

version = "1.0.0"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="shield_2018",
    version=version,
    description=
    "Tools used in the Chung Lab's Shield paper, 2018",
    long_description=long_description,
    install_requires=[
        "phathom"
    ],
    author="Kwanghun Chung Lab",
    packages=["shield_2018"],
    entry_points={ 'console_scripts': [
        'shield-2018-segment=shield_2018.segmentation:main',
        'download-assets=shield_2018.download_assets:main'
    ]},
    url="https://github.com/chunglabmit/shield_2018",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)