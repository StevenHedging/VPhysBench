.PHONY: test legacy-test smoke

test:
	PYTHONPATH=src python3 -m unittest tests.test_six_scene_dataset_v8 -v

# Historical suites intentionally retain old Dataset IDs and may require media
# paths that no longer exist in the current asset layout.
legacy-test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	@set -eu; \
	smoke_root="$$(mktemp -d -t physbench-smoke.XXXXXX)"; \
	trap 'rm -r "$$smoke_root"' EXIT; \
	PYTHONPATH=src python3 -m physbench baseline init smoke_submission \
		--backend submission --root "$$smoke_root/baselines" >/dev/null; \
	PYTHONPATH=src python3 -m physbench atomic-run \
		--dataset datasets/releases/10.0.0/dataset.json \
		--task tasks/official/five_scene_direct_eval.json \
		--baseline "$$smoke_root/baselines/smoke_submission" \
		--case-id circular_r1_silver02cm_img_0370 \
		--run-id smoke \
		--output-root "$$smoke_root/runs" >/dev/null
