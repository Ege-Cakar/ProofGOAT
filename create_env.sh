#!/bin/bash
pyenv virtualenv 3.11.0 .venv
pyenv activate 3.11.0/envs/.venv
python --version
pip install -r requirements.txt