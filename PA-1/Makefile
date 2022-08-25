greet:
	echo "Welcome to Makefile"

makeEnv:
	virtualenv venv

deleteEnv:
	rm -r venv

installReq: requirements.txt
	pip install -r requirements.txt

UpdReq: 
	pip freeze > requirements.txt

convertNotebook: 
	jupyter nbconvert Q*.ipynb --to python

cleanScript:
	rm *.py
