To install virtualenv use following command:

```pip install virtualenv```

To build venv

make makeEnv


To activate virtual environment use following command:

# In cmd.exe
venv\Scripts\activate.bat
# In PowerShell
venv\Scripts\Activate.ps1
# In Linux
source venv/bin/activate

To install requirements.txt use following command after activating venv:

pip install -r requirements.txt

or

make installReq

To deactivate use:

deactivate

To remove virtual environment use following command:

make deleteEnv

