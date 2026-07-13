# Makefile for TrulyIAS Scheduling Assistant

.PHONY: install setup test run clean lint format

install:
	pip install -r requirements.txt

setup:
	python scripts/setup_db.py

test:
	pytest tests/ -v

run:
	streamlit run streamlit_app.py

lint:
	ruff check src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.db" -delete
