SILENT:
.PHONY: help docs install-dependencies create-venv setup
.DEFAULT_GOAL = help

PYTHON := py -3.9
VENV_NAME := .venv
VENV_PYTHON := $(VENV_NAME)/Scripts/python.exe
ACTIVATE_VENV := $(VENV_NAME)/Scripts/activate.bat

COLOR_RESET = \033[0m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
PROJECT_NAME = pyAutoSummarizer $(PWD)
KERNEL_NAME := $(shell uname -s)

## Prints this help
help:
	@printf "${COLOR_YELLOW}\n${PROJECT_NAME}\n\n${COLOR_RESET}"
	@awk '/^[a-zA-Z\-\0-9\.%]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "${COLOR_GREEN}$$ make %s${COLOR_RESET} %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
	@printf "\n"

create-venv:
	@if [ -d $(VENV_NAME) ]; then \
		echo "Venv já existe..."; \
	else \
		echo "Criando novo venv..."; \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi; 

install-dependencies:
	$(VENV_PYTHON) -m pip install --upgrade pip; \
	$(VENV_PYTHON) -m pip install -e .

## initial setup to repo 
setup:
	@make create-venv
	@make install-dependencies

## 📦 Gera distribuição do pacote (wheel + source)
build:
	@$(VENV_PYTHON) setup.py sdist bdist_wheel