.PHONY: setup run build-vectors clean

setup:
	python -m venv venv
	venv\Scripts\activate || source venv/bin/activate
	pip install -r requirements.txt

run:
	python -m streamlit run app.py

build-vectors:
	python scripts/build_vectorstore.py

clean:
	rm -rf data/vector_db