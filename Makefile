PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup setup-gpu smoke test test-degrade manifest-demo encode-demo probe-demo clean

# Mac dev path: torch-free. The GPU box additionally runs `make setup-gpu`.
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"
	@echo "✓ setup complete ($$($(PY) --version))"

# GPU box only (Spain). Fails loudly on a machine without torch wheels — by design.
setup-gpu:
	$(PIP) install -e ".[gpu]"
	@echo "✓ gpu backends installed"

smoke:
	$(PY) -m pilot0.smoke

test:
	$(PY) -m pytest -q

# Phase-1 acceptance: measured-vs-target table across the degradation grid.
test-degrade:
	$(PY) -m pilot0.degrade.audit

# Phase-2 demo: synth corpus → preflight → manifest → summary.
manifest-demo:
	$(PY) -m pilot0.corpus.audit

# Phase-3 demo: synth corpus → manifest → fake-encode → cache summary + resume.
encode-demo:
	$(PY) -m pilot0.encode.audit

# Phase-4 demo: synth corpus → encode → Gate-1 table (fake codecs vs real floor).
probe-demo:
	$(PY) -m pilot0.probes.audit

clean:
	rm -rf $(VENV) src/*.egg-info reports .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
