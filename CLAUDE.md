# Project Overview

This repository provides guidance to Claude Code (claude.ai/code) to perform autoresearch machine learning to compete in Kaggle competitions.

# RULES
- Use uv to manage python environment, create .venv if not exists from requirements.txt
- Activate the .venv environment before performing any python related works.
- DO NOT USE, MODIFY, INSTALL or TOUCH the system python, typically located at /usr/bin/python3. Instead MUST use python in .venv when perform any python related works.
- From now on if you need to update the python packages, update the requirements.txt first, then run uv pip to update the packages.
- Work ONLY in the current project root directory autoresearch-agents-kaggle, Do NOT go to other directories unless explicitly allowed. 