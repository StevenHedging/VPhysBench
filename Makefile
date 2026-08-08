.PHONY: test full-test smoke release-check

test:
	PYTHONPATH=src:tests:. python3 -m unittest tests.test_current_dataset tests.test_single_current_physics_v13 -v

full-test:
	PYTHONPATH=src:tests:. python3 -m unittest discover -s tests -v

smoke:
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
