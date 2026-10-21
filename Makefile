PYTHON ?= python

.PHONY: data test eval-mock

data:
	$(PYTHON) data/make_test_data.py
	bash data/make_variant_data.sh

test:
	pytest -q

eval-mock:
	gta eval --mock
