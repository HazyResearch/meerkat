port ?= "8000"

autoformat:
	black meerkat/ tests/
	isort --atomic meerkat/ tests/
	docformatter --in-place --recursive meerkat tests

lint:
	isort -c meerkat/ tests/
	black meerkat/ tests/ --check
	flake8 meerkat/ tests/

test:
	pytest tests/

test-basic:
	set -e
	python -c "import meerkat as mk"
	python -c "import meerkat.version as mversion"

test-cov:
	pytest --cov=./ --cov-report=xml

test-interactive-install:
	set -e
	mk install --no-run-dev

docs:
	set -e
	rm -rf docs/source/apidocs/generated
	python docs/source/rst_gen.py
	sphinx-build -b html docs/source/ docs/build/html/

docs-check:
	python docs/source/rst_gen.py
	rm -rf docs/source/apidocs/generated
	sphinx-build -b html docs/source/ docs/build/html/ -W

livedocs:
	rm -rf docs/source/apidocs/generated
	python docs/source/rst_gen.py
	SPHINX_LIVEDOCS=true sphinx-autobuild -b html docs/source/ docs/build/html/ --port=${port}


website:
	set -e
	make docs -B
	rm -rf website/static/docs
	mkdir -p website/static/docs
	cp -r docs/build/html/* website/static/docs/
	python website/reroute.py
	cd website && npm run build


dev:
	pip install black==22.12.0 isort flake8 docformatter pytest-cov sphinx-rtd-theme nbsphinx recommonmark pre-commit sphinx-panels jupyter-sphinx pydata-sphinx-theme sphinx-autobuild sphinx-toolbox sphinx-copybutton sphinx_design

all: autoformat lint docs test
