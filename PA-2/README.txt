To install the virtual environment

run the following commands:-
1) virtualenv venv
2) pip install -r requirements.txt

To activate the virtual environment
# In cmd.exe
venv\Scripts\activate.bat

To run the assignment use the following commands:-
python Q1.py
python Q2.py

To deactivate the virtual environment
# In cmd.exe
venv\Scripts\deactivate.bat

To save the requirements.txt

pip freeze > requirements.txt

To convert the Notebook to a python script

jupyter nbconvert Q*.ipynb --to python