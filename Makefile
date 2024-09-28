# Makefile

# Variables
VENV := venv
PYTHON := python3
PORT := 3000

# Command: make install
install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

# Command: make run
run:
	. $(VENV)/bin/activate && FLASK_APP=web_server.py flask run --host=0.0.0.0 --port=$(PORT)

# Clean up virtual environment
clean:
	rm -rf $(VENV)
