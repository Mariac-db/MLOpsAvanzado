#when you run 'make' in the terminal, it will run the help command
#use '@' in front of command so that it won't show up in the terminal when you run it

.DEFAULT_GOAL = help 

install: #install dependencies 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate: # activa poetry env
	@echo "Activating virtual environment..."
	poetry shell

initializate_git: #initilizaze git
	@echo "Initializing git..."
	git init

setup: initializate_git