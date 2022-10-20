# use a virtual environment
python3 -m venv whatIf-Env
source ./whatIf-Env/bin/activate
python -m pip install -r requirements.txt

# get test coverage
python -m coverage run -m pytest 
python -m coverage report -m

# run the tests
python -m pytest 

# run the tests without capturing output (good for ipdb)
python -m pytest -s

# run a specific test file
python -m pytest tests/test_snn_refactoring_prototyping.py

# run a specific test function
python -m pytest tests/test_snn_refactoring_prototyping.py::test_snn
