.PHONY: test data-test full-test smoke smoke-interface smoke-data release-check release-archive-check

test:
	PYTHONPATH=src:tests:. python3 -m unittest \
		tests.test_current_dataset \
		tests.test_huggingface_dataset_binding \
		tests.test_dataset_hub_cli \
		tests.test_managed_baselines \
		tests.test_clean_baseline_smoke \
		tests.test_release_audit \
		tests.test_release_documentation \
		tests.test_repository_portability \
		tests.test_release_archive -v

data-test:
	PYTHONPATH=src:tests:. python3 -m physbench validate-dataset \
		--dataset datasets/releases/13.0.0/dataset.json --check-assets

full-test:
	PYTHONPATH=src:tests:. python3 -m unittest discover -s tests -v

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
		--dataset datasets/releases/13.0.0/dataset.json \
		--task tasks/official/five_scene_direct_eval.json \
		--baseline "$$smoke_root/baselines/smoke_submission" \
		--case-id circular_r1_silver02cm_img_0370 \
		--run-id smoke \
		--output-root "$$smoke_root/runs" >/dev/null

release-check:
	PYTHONPATH=src:tests:. python3 scripts/release_audit.py .

release-archive-check:
	PYTHONPATH=src:tests:. python3 scripts/verify_release_archive.py
