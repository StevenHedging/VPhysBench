.PHONY: test test-interface test-evaluation data-test smoke smoke-interface smoke-data release-check release-archive-check portable-release-check reference-observation-audit evaluation-preflight

test: test-interface

test-interface:
	PYTHONPATH=src:tests:. python3 -m unittest \
		tests.test_lightweight_import_boundaries \
		tests.test_current_dataset \
		tests.test_release_v1_contract \
		tests.test_dataset_contract_v4 \
		tests.test_huggingface_dataset_binding \
		tests.test_environment_doctor \
		tests.test_dataset_hub_cli \
		tests.test_dataset_distribution \
		tests.test_managed_baselines \
		tests.test_clean_baseline_smoke \
		tests.test_release_audit \
		tests.test_release_documentation \
		tests.test_repository_portability \
		tests.test_release_archive \
		tests.test_validation -v

test-evaluation:
	PYTHONPATH=src:tests:. python3 -m unittest \
		tests.test_evaluation_protocol_v1 \
		tests.test_scene_evaluation \
		tests.test_evaluation_preflight \
		tests.test_csti_case_integration -v

data-test:
	PYTHONPATH=src:tests:. python3 -m physbench validate-dataset \
		--dataset datasets/releases/14.0.0/dataset.json --check-assets

smoke: smoke-interface

smoke-interface:
	PYTHONPATH=src:tests:. python3 scripts/smoke_custom_baseline.py --metadata-only

smoke-data:
	@set -eu; \
	smoke_root="$$(mktemp -d -t physbench-smoke.XXXXXX)"; \
	trap 'rm -r "$$smoke_root"' EXIT; \
	PYTHONPATH=src python3 -m physbench baseline init smoke_submission \
		--backend submission --root "$$smoke_root/baselines" >/dev/null; \
	PYTHONPATH=src python3 -m physbench atomic-run \
		--dataset datasets/releases/14.0.0/dataset.json \
		--task tasks/official/six_scene_direct_eval_v1.json \
		--baseline "$$smoke_root/baselines/smoke_submission" \
		--case-id circular_r1_silver02cm_img_0370 \
		--run-id smoke \
		--output-root "$$smoke_root/runs" >/dev/null

release-check:
	PYTHONPATH=src:tests:. python3 scripts/release_audit.py .

release-archive-check:
	PYTHONPATH=src:tests:. python3 scripts/verify_release_archive.py

portable-release-check:
	@set -eu; \
	bootstrap_python="$${VPHYSBENCH_BOOTSTRAP_PYTHON:-}"; \
	if [ -z "$$bootstrap_python" ]; then \
		for candidate in python3.12 python3.11 python3; do \
			if command -v "$$candidate" >/dev/null 2>&1 \
				&& "$$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then \
				bootstrap_python="$$candidate"; break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$bootstrap_python" ]; then \
		echo "no Python 3.11+ interpreter found for bootstrap dry-run" >&2; exit 1; \
	fi; \
	PYTHONPATH=src:tests:. "$$bootstrap_python" scripts/verify_release_archive.py; \
	gate_root="$$(mktemp -d -t vphysbench-portable.XXXXXX)"; \
	trap 'rm -rf "$$gate_root"' EXIT; \
	(cd "$$gate_root" && VPHYSBENCH_BOOTSTRAP_PYTHON="$$bootstrap_python" \
		bash "$(CURDIR)/scripts/bootstrap_env.sh" --profile metadata --dry-run)

reference-observation-audit:
	PYTHONPATH=src:tests:. python3 scripts/reference_observations/audit_v14.py \
		--dataset datasets/releases/14.0.0/dataset.json \
		--output .local/reference_observation_curation/full/diagnostics.jsonl \
		--render-root .local/reference_observation_curation/full/review \
		--no-install

evaluation-preflight:
	PYTHONPATH=src:tests:. python3 scripts/evaluation_preflight.py \
		--dataset datasets/releases/14.0.0/dataset.json \
		--task tasks/official/six_scene_train_six_scene_eval_v1.json \
		--output .local/evaluation_preflight/six_scene_train_six_scene_eval_v1
