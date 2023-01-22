
all: test

.PHONY: mypy
mypy:
	mypy pack

.PHONY: test
test:
	pytest -x

.PHONY: lint
lint:
	flake8 pack


.PHONY: clean
clean:
	rm -Rf dist *.egg-info build
