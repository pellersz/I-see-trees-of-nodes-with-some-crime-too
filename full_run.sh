#!/bin/bash 

uv run setup.py build_ext --inplace
echo " "
uv run main.py
