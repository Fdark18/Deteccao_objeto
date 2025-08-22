# Makefile para Sistema de Detecção de Dispositivos
# Comandos úteis para desenvolvimento e execução

.PHONY: help install install-dev run test lint format type-check clean setup

# Configurações
PYTHON := python
PIP := pip
POETRY := poetry

help:
	@echo "Comandos disponíveis:"
	@echo "  install      - Instala dependências de produção"
	@echo "  install-dev  - Instala todas as dependências incluindo dev"
	@echo "  run          - Executa o detector"
	@echo "  test         - Executa testes"
	@echo "  lint         - Executa linting (flake8, pylint)"
	@echo "  format       - Formata código (black, isort)"
	@echo "  type-check   - Verifica tipos (mypy)"
	@echo "  clean        - Remove arquivos temporários"
	@echo "  setup        - Configuração inicial completa"

install:
	@echo "Instalando dependências de produção..."
	$(PIP) install ultralytics opencv-python numpy torch torchvision pillow

install-dev:
	@echo "Instalando todas as dependências..."
	$(PIP) install -r requirements.txt

install-poetry:
	@echo "Instalando com Poetry..."
	$(POETRY) install

run:
	@echo "Executando detector de dispositivos..."
	$(PYTHON) detector_celular_strict.py

test:
	@echo "Executando testes..."
	$(PYTHON) -m pytest tests/ -v

lint:
	@echo "Executando verificações de código..."
	$(PYTHON) -m flake8 detector_celular_strict.py
	$(PYTHON) -m pylint detector_celular_strict.py

format:
	@echo "Formatando código..."
	$(PYTHON) -m black detector_celular_strict.py
	$(PYTHON) -m isort detector_celular_strict.py

type-check:
	@echo "Verificando tipos..."
	$(PYTHON) -m mypy detector_celular_strict.py

clean:
	@echo "Removendo arquivos temporários..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/

setup: clean install-dev
	@echo "Configuração inicial completa!"
	@echo "Execute 'make run' para iniciar o detector"

# Comandos específicos para VS Code
vscode-setup:
	@echo "Configurando VS Code..."
	mkdir -p .vscode
	@echo "Criando settings.json para VS Code..."

# Comandos para Docker (opcional)
docker-build:
	@echo "Construindo imagem Docker..."
	docker build -t detector-dispositivos .

docker-run:
	@echo "Executando container Docker..."
	docker run --device=/dev/video0 -it detector-dispositivos

# Verificação de sistema
check-system:
	@echo "Verificando sistema..."
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "OpenCV disponível: $$($(PYTHON) -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'NÃO INSTALADO')"
	@echo "PyTorch disponível: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NÃO INSTALADO')"
	@echo "Ultralytics disponível: $$($(PYTHON) -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'NÃO INSTALADO')"

# Para Windows (PowerShell)
install-windows:
	powershell -Command "pip install ultralytics opencv-python numpy torch torchvision pillow"

run-windows:
	powershell -Command "python detector_celular_strict.py"