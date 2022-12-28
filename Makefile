autoformat:
	black meerkat/ tests/
	isort --atomic meerkat/ tests/
	docformatter --in-place --recursive meerkat tests

lint:
	isort -c meerkat/ tests/
	black meerkat/ tests/ --check
	flake8 meerkat/ tests/

test:
	pytest

test-basic:
	set -e
	python -c "import meerkat as mk"
	python -c "import meerkat.version as mversion"

test-cov:
	pytest --cov=./ --cov-report=xml

docs:
	sphinx-build -b html docs/source/ docs/build/html/

docs-check:
	sphinx-build -b html docs/source/ docs/build/html/ -W

livedocs:
	sphinx-autobuild -b html docs/source/ docs/build/html/

dev:
	pip install black==22.12.0 isort flake8 docformatter pytest-cov sphinx-rtd-theme nbsphinx recommonmark pre-commit sphinx-panels jupyter-sphinx pydata-sphinx-theme sphinx-autobuild sphinx-toolbox sphinx-copybutton

all: autoformat lint docs test
