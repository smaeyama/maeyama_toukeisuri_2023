#!/bin/sh

#find ./data/ -type f -name *.dat | xargs rm -rf
find . -type d -name .ipynb_checkpoints | xargs rm -rf
find . -type d -name __pycache__ | xargs rm -rf

#jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
jupyter nbconvert --to python */*.ipynb


