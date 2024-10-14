all: install run
install :
	pip install conllu
	pip install pyconll
	pip install FastText
	pip install keras
run :
	python3 english_to_hindi_translator.py