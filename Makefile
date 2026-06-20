SHELL := /bin/bash

PYTHON_VERSION := 3.11.9
VENV := airbnb-ve
PYTHON := $(VENV)/Scripts/python
KERNEL_NAME := airbnb-reviewer
NOTEBOOK := notebook/Ejercicio_1_Gemini_Groq.ipynb
export XDG_CACHE_HOME := $(CURDIR)/.cache
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache

.PHONY: help setup venv sync kernel check notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup    Crea el entorno completo y registra el kernel de Jupyter"
	@echo "  make check    Verifica imports y versiones principales"
	@echo "  make notebook Abre JupyterLab con el entorno del proyecto"
	@echo "  make clean    Elimina el entorno virtual"

setup: venv sync kernel check
	@echo "Entorno listo. Usa el kernel '$(KERNEL_NAME)' en $(NOTEBOOK)."

venv:
	@test -d "$(VENV)" || python -m venv "$(VENV)"

sync: venv
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt

kernel:
	@$(PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"

check:
	@mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"
	@$(PYTHON) -c 'import importlib, platform; from importlib.metadata import version; modules = {"google-genai": "google.genai", "pandas": "pandas", "python-dotenv": "dotenv", "groq": "groq", "IPython": "IPython", "tensorflow": "tensorflow", "transformers": "transformers", "tf-keras": "tf_keras", "accelerate": "accelerate", "bitsandbytes": "bitsandbytes", "huggingface-hub": "huggingface_hub", "sentencepiece": "sentencepiece", "ipykernel": "ipykernel"}; [importlib.import_module(module) for module in modules.values()]; print(f"Python: {platform.python_version()}"); [print(f"{name}: {version(name)}") for name in modules]'

notebook:
	@$(PYTHON) -m jupyter lab $(NOTEBOOK)

clean:
	@rm -rf "$(VENV)"
