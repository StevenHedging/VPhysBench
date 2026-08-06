from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace

from _paths import ROOT
from baselines.wan22_symbol_value_cross_attention.adapter import (
    SymbolValueRegistry,
)
import torch
from torch import nn

from physbench.baselines.wan22_symbol_value_model import (
    NUMERIC_FEATURE_NAMES,
    SymbolValueConditioner,
    SYMBOL_VALUE_CHECKPOINT_PREFIX,
    load_combined_symbol_value_checkpoint,
    load_symbol_value_conditioner_checkpoint,
    numeric_features,
    pool_symbol_embeddings,
    symbol_value_inference_conditioning,
)
from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.tasks import load_task
from physbench.evaluation import load_evaluation_protocol


REGISTRY = (
    ROOT
    / "baselines"
    / "wan22_symbol_value_cross_attention"
    / "quantity_registry.json"
)
BASELINE = (
    ROOT
    / "baselines"
    / "wan22_symbol_value_cross_attention"
    / "baseline.json"
)
TASK = ROOT / "tasks" / "experiments" / "seven_scene_symbol_value_finetune_eval.json"


def _small_conditioner_config() -> dict:
    return {
        "numeric_feature_size": len(NUMERIC_FEATURE_NAMES),
        "numeric_hidden_size": 12,
        "numeric_embedding_size": 8,
        "dimension_size": 7,
        "dimension_hidden_size": 8,
        "dimension_embedding_size": 6,
        "unit_count": 9,
        "unit_embedding_size": 4,
        "value_fusion_hidden_size": 16,
        "text_hidden_size": 8,
        "attention_hidden_size": 4,
        "attention_heads": 2,
        "dropout": 0.0,
        "residual_gate_init": 1.0,
    }


def _records() -> list[dict]:
    return [
        {
            "name": "length",
            "symbol": "x_0",
            "si_value": 0.1,
            "dimension": [1, 0, 0, 0, 0, 0, 0],
            "unit_id": 1,
            "raw_value": 0.1,
            "raw_unit": "m",
            "canonical_si_unit": "m",
        },
        {
            "name": "mass",
            "symbol": "m",
            "si_value": 0.5,
            "dimension": [0, 1, 0, 0, 0, 0, 0],
            "unit_id": 2,
            "raw_value": 0.5,
            "raw_unit": "kg",
            "canonical_si_unit": "kg",
        },
    ]


def _fake_safetensors_modules(state: dict[str, torch.Tensor]) -> dict[str, ModuleType]:
    package = ModuleType("safetensors")
    torch_module = ModuleType("safetensors.torch")
    torch_module.load_file = lambda _path, device="cpu": {
        key: value.to(device) for key, value in state.items()
    }
    package.torch = torch_module
    return {"safetensors": package, "safetensors.torch": torch_module}


class _SymbolTokenizer:
    pad_token_id = 0

    def __call__(self, symbols, **kwargs):
        del kwargs
        mapping = {"x_0": [1, 2], "m": [3]}
        rows = [mapping[symbol] for symbol in symbols]
        width = max(map(len, rows))
        ids = [row + [0] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _SymbolWrapper:
    def __init__(self):
        self.tokenizer = _SymbolTokenizer()


class SymbolValueModelTests(unittest.TestCase):
    def test_numeric_features_are_hand_derived(self) -> None:
        self.assertEqual(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            numeric_features(0.0),
        )
        ten = numeric_features(10.0)
        self.assertEqual(0.0, ten[0])
        self.assertEqual(1.0, ten[1])
        self.assertAlmostEqual(math.log(11.0), ten[2])
        self.assertAlmostEqual(0.1, ten[5])
        self.assertAlmostEqual(1.0 / 12.0, ten[6])
        self.assertAlmostEqual(1.0 / 11.0, ten[7])
        negative = numeric_features(-0.1)
        self.assertEqual(-1.0, negative[1])
        self.assertLess(negative[3], 0.0)
        self.assertLess(negative[5], 0.0)

    def test_pool_symbol_embeddings_means_only_non_padding_subwords(self) -> None:
        embedding = nn.Embedding(4, 3, padding_idx=0)
        with torch.no_grad():
            embedding.weight.copy_(torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [2.0, 0.0, 0.0],
            ]))

        pooled, audit = pool_symbol_embeddings(
            _SymbolWrapper(),
            embedding,
            _records(),
        )

        self.assertTrue(torch.equal(torch.tensor([2.0, 3.0, 4.0]), pooled[0]))
        self.assertTrue(torch.equal(torch.tensor([2.0, 0.0, 0.0]), pooled[1]))
        self.assertEqual([1, 2], audit[0]["symbol_token_ids"])
        self.assertEqual([3], audit[1]["symbol_token_ids"])
        self.assertFalse(pooled.requires_grad)

    def test_conditioner_builds_one_additive_token_per_record(self) -> None:
        torch.manual_seed(7)
        conditioner = SymbolValueConditioner(_small_conditioner_config())
        symbols = torch.randn(2, 8)

        tokens = conditioner.build_physics_tokens(symbols, _records())

        self.assertEqual((2, 8), tuple(tokens.shape))
        self.assertTrue(bool(torch.isfinite(tokens).all()))
        self.assertTrue(torch.allclose(
            tokens,
            conditioner.fusion_norm(
                conditioner.symbol_norm(symbols)
                + conditioner.encode_value_records(_records())
            ),
        ))

    def test_cross_attention_changes_valid_text_and_zeros_padding(self) -> None:
        torch.manual_seed(11)
        conditioner = SymbolValueConditioner(_small_conditioner_config())
        conditioner.eval()
        context = torch.randn(1, 4, 8)
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        symbols = torch.randn(2, 8)

        output, attention = conditioner(
            context,
            mask,
            symbols,
            _records(),
        )

        self.assertEqual((1, 4, 8), tuple(output.shape))
        self.assertEqual((1, 2, 4, 2), tuple(attention.shape))
        self.assertFalse(torch.allclose(output[:, :3], context[:, :3]))
        self.assertTrue(torch.equal(torch.zeros(1, 8), output[:, 3]))

    def test_inference_binding_is_positive_branch_only_and_restores_state(self) -> None:
        pipe = SimpleNamespace(existing="kept")

        with symbol_value_inference_conditioning(pipe, _records()):
            self.assertEqual(
                _records(),
                pipe._active_symbol_value_conditioning["positive_quantities"],
            )
            self.assertIsNone(
                pipe._active_symbol_value_conditioning["negative_quantities"]
            )

        self.assertFalse(hasattr(pipe, "_active_symbol_value_conditioning"))


class SymbolValueCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.conditioner = SymbolValueConditioner(_small_conditioner_config())
        self.conditioner_state = {
            f"{SYMBOL_VALUE_CHECKPOINT_PREFIX}{key}": value.detach().clone()
            for key, value in self.conditioner.state_dict().items()
        }

    def test_conditioner_checkpoint_requires_exact_finite_topology(self) -> None:
        with patch.dict(
            "sys.modules",
            _fake_safetensors_modules(self.conditioner_state),
        ):
            result = load_symbol_value_conditioner_checkpoint(
                self.conditioner,
                "fixture.safetensors",
                required=True,
            )
        self.assertTrue(result["loaded"])
        self.assertEqual(len(self.conditioner.state_dict()), result["tensor_count"])

        missing = dict(self.conditioner_state)
        missing.pop(next(iter(missing)))
        with patch.dict("sys.modules", _fake_safetensors_modules(missing)):
            with self.assertRaisesRegex(ValueError, "topology"):
                load_symbol_value_conditioner_checkpoint(
                    self.conditioner,
                    "fixture.safetensors",
                    required=True,
                )

        non_finite = dict(self.conditioner_state)
        key = next(iter(non_finite))
        non_finite[key] = non_finite[key].clone()
        non_finite[key].view(-1)[0] = float("nan")
        with patch.dict("sys.modules", _fake_safetensors_modules(non_finite)):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_symbol_value_conditioner_checkpoint(
                    self.conditioner,
                    "fixture.safetensors",
                    required=True,
                )

    def test_combined_checkpoint_validates_both_states_before_mutation(self) -> None:
        state = {**self.conditioner_state, "fake_lora": torch.ones(1)}
        pipe = SimpleNamespace(
            dit=nn.Linear(1, 1),
            load_lora=lambda *args, **kwargs: setattr(pipe, "loaded", kwargs),
        )
        validator = (
            "physbench.baselines.wan22_quantity_model."
            "_validate_dit_lora_state_dict"
        )
        with (
            patch.dict("sys.modules", _fake_safetensors_modules(state)),
            patch(validator, return_value=("layer",)) as validate_lora,
        ):
            result = load_combined_symbol_value_checkpoint(
                pipe,
                self.conditioner,
                "fixture.safetensors",
                lora_alpha=1.0,
            )
        validate_lora.assert_called_once()
        self.assertEqual(1, result["dit_lora_tensor_count"])
        self.assertTrue(hasattr(pipe, "loaded"))

        invalid = dict(state)
        invalid.pop(next(iter(self.conditioner_state)))
        untouched = SimpleNamespace(
            dit=nn.Linear(1, 1),
            load_lora=lambda *args, **kwargs: setattr(untouched, "loaded", True),
        )
        with (
            patch.dict("sys.modules", _fake_safetensors_modules(invalid)),
            patch(validator) as validate_lora,
        ):
            with self.assertRaisesRegex(ValueError, "topology"):
                load_combined_symbol_value_checkpoint(
                    untouched,
                    self.conditioner,
                    "fixture.safetensors",
                    lora_alpha=1.0,
                )
        validate_lora.assert_not_called()
        self.assertFalse(hasattr(untouched, "loaded"))


class SymbolValueBaselineIntegrationTests(unittest.TestCase):
    def test_bundle_is_discovered_with_sealed_training_recipe(self) -> None:
        baseline_id = "wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1"
        discovered = discover_baseline_bundles()
        self.assertIn(baseline_id, discovered)
        self.assertEqual(BASELINE, discovered[baseline_id])

        bundle = load_baseline_bundle(BASELINE)
        self.assertEqual("1.0.0", bundle.value["baseline_version"])
        self.assertEqual(
            ["symbol_value_cross_attention_v1"],
            bundle.value["input_policy"]["physics"]["representations"],
        )
        trainer = bundle.value["trainer"]["config"]
        self.assertEqual(32, trainer["rank"])
        self.assertEqual(1, trainer["dataset_repeat"])
        self.assertEqual(2, trainer["num_epochs"])
        self.assertEqual(42, trainer["seed"])
        conditioner = bundle.value["model"]["symbol_value_conditioner"]
        self.assertEqual(4096, conditioner["text_hidden_size"])
        self.assertEqual(512, conditioner["attention_hidden_size"])
        self.assertEqual(8, conditioner["attention_heads"])

    def test_real_dataset_cases_adapt_without_changing_captions(self) -> None:
        dataset = load_dataset(LATEST_DATASET, check_assets=False)
        bundle = load_baseline_bundle(BASELINE)
        adapter = load_baseline_plugin(bundle).task_builder.data_adapter

        for case in dataset.cases:
            adaptation = adapter.adapt_case(case, role="eval")
            self.assertEqual(
                case["text"]["prompt"].strip(),
                adaptation["native_inputs"]["text"]["prompt"],
            )
            quantities = adaptation["native_inputs"]["physics"]["quantities"]
            self.assertTrue(quantities, case["case_id"])
            self.assertTrue(all(item["symbol"] in case["text"]["prompt"]
                                for item in quantities))

    def test_runtime_entrypoints_are_declared_and_present(self) -> None:
        expected = [
            ROOT / "scripts" / "wan22_symbol_value_train.py",
            ROOT / "scripts" / "train_wan22_symbol_value.sh",
            ROOT / "scripts" / "wan22_symbol_value_generate.py",
            ROOT / "scripts" / "wan22_symbol_value_generate_batch.py",
            ROOT / "src" / "physbench" / "baselines" / "wan22_symbol_value.py",
            ROOT / "src" / "physbench" / "baseline_plugins" / "wan22_symbol_value.py",
            ROOT / "src" / "physbench" / "baseline_runtime" / "drivers" / "wan22_symbol_value.py",
        ]
        self.assertTrue(all(path.is_file() for path in expected))


class SevenSceneTaskTests(unittest.TestCase):
    def test_task_compiles_every_view_a_train_and_test_case(self) -> None:
        dataset = load_dataset(LATEST_DATASET, check_assets=False)
        task = load_task(TASK)
        plugin = load_baseline_plugin(load_baseline_bundle(BASELINE))

        instance = plugin.task_builder.build(dataset, task)

        expected = {
            "pendulum": (80, 20),
            "collision_1d": (310, 20),
            "inclined_plane_slide": (80, 15),
            "uniform_circular_motion": (30, 6),
            "parabolic_motion": (82, 15),
            "push_bottle": (127, 14),
            "vertical_spring_oscillator": (97, 20),
        }
        train_ids = set(instance.canonical_plan.train_case_ids)
        self.assertEqual(806, len(train_ids))
        self.assertEqual(110, len(instance.canonical_plan.jobs))
        self.assertFalse(train_ids & {
            job["case_id"] for job in instance.canonical_plan.jobs
        })
        by_id = {case["case_id"]: case for case in dataset.cases}
        for scene_id, (train_count, test_count) in expected.items():
            self.assertEqual(
                train_count,
                sum(by_id[case_id]["scene_id"] == scene_id
                    for case_id in train_ids),
            )
            self.assertEqual(
                test_count,
                sum(job["scene_id"] == scene_id
                    for job in instance.canonical_plan.jobs),
            )

    def test_protocol_scores_five_scenes_and_marks_two_unsupported(self) -> None:
        protocol = load_evaluation_protocol(
            "scene_default_v10_seven_scene_unscored"
        )

        self.assertEqual(
            {"push_bottle", "vertical_spring_oscillator"},
            {
                scene_id for scene_id, config in protocol["scenes"].items()
                if config["type"] == "unsupported"
            },
        )
        self.assertEqual("pendulum_state_v7", protocol["scenes"]["pendulum"]["type"])


def _case() -> dict:
    return {
        "case_id": "spring_fixture",
        "scene_id": "vertical_spring_oscillator",
        "text": {
            "prompt": (
                "An oscillator of mass m and radius r is attached to a spring "
                "of stiffness k and natural length L_0 under gravity g. Its "
                "initial displacement x_0 is above equilibrium."
            )
        },
        "physics": {
            "objects": {
                "object_1": {
                    "initial_displacement": {
                        "symbol": "x_0",
                        "value": 0.1,
                        "unit": "m",
                    },
                    "mass": {
                        "symbol": "m",
                        "value": 0.5156,
                        "unit": "kg",
                    },
                    "radius": {
                        "symbol": "r",
                        "value": 0.025,
                        "unit": "m",
                    },
                }
            },
            "environment": {
                "gravity_acceleration": {
                    "symbol": "g",
                    "value": 9.80665,
                    "unit": "m/s^2",
                },
                "natural_spring_length": {
                    "symbol": "L_0",
                    "value": 0.068,
                    "unit": "m",
                },
                "spring_stiffness": {
                    "symbol": "k",
                    "value": 32.6213467096774,
                    "unit": "N/m",
                },
            },
        },
    }


class SymbolValueRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SymbolValueRegistry(REGISTRY)

    def test_render_preserves_prompt_and_emits_si_symbol_value_records(self) -> None:
        case = _case()

        prompt, records, used = self.registry.render(
            case,
            case["text"]["prompt"],
        )

        self.assertEqual(case["text"]["prompt"], prompt)
        self.assertEqual(6, len(records))
        displacement = next(
            item for item in records
            if item["name"] == "initial_displacement"
        )
        self.assertEqual("x_0", displacement["symbol"])
        self.assertEqual(0.1, displacement["si_value"])
        self.assertEqual([1, 0, 0, 0, 0, 0, 0], displacement["dimension"])
        self.assertIsInstance(displacement["unit_id"], int)
        self.assertEqual(
            {item["name"] for item in records},
            set(used),
        )
        self.assertNotIn("0.100", prompt)

    def test_si_conversion_uses_registry_scale(self) -> None:
        case = {
            "case_id": "pendulum_fixture",
            "scene_id": "pendulum",
            "text": {
                "prompt": (
                    "A bob of mass m and radius r is released at angle θ_0 "
                    "on a string of length l_s."
                )
            },
            "physics": {
                "objects": {
                    "object_1": {
                        "radius": {
                            "symbol": "r", "value": 0.01, "unit": "m"
                        },
                        "initial_angle": {
                            "symbol": "θ_0", "value": 30.0, "unit": "deg"
                        },
                        "mass": {
                            "symbol": "m", "value": 0.03, "unit": "kg"
                        },
                    }
                },
                "environment": {
                    "string_length": {
                        "symbol": "l_s", "value": 0.5, "unit": "m"
                    }
                },
            },
        }

        _, records, _ = self.registry.render(
            case,
            case["text"]["prompt"],
        )

        angle = next(item for item in records if item["name"] == "initial_angle")
        self.assertAlmostEqual(math.pi / 6.0, angle["si_value"])
        self.assertEqual("rad", angle["canonical_si_unit"])

    def test_missing_symbol_in_prompt_is_rejected(self) -> None:
        case = _case()
        case["text"]["prompt"] = case["text"]["prompt"].replace("x_0", "x")

        with self.assertRaisesRegex(ValueError, "symbol.*x_0.*absent"):
            self.registry.render(case, case["text"]["prompt"])

    def test_unexpected_unit_is_rejected(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"]["unit"] = "g"

        with self.assertRaisesRegex(ValueError, "unit.*g.*kg"):
            self.registry.render(case, case["text"]["prompt"])

    def test_non_finite_value_is_rejected(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"]["value"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            self.registry.render(case, case["text"]["prompt"])

    def test_temporal_quantity_is_not_collapsed_to_a_scalar(self) -> None:
        case = _case()
        case["physics"]["objects"]["object_1"]["mass"] = {
            "symbol": "m",
            "samples": [{"time": 0.0, "value": 0.5}],
            "time_unit": "s",
            "unit": "kg",
        }

        with self.assertRaisesRegex(ValueError, "scalar"):
            self.registry.render(case, case["text"]["prompt"])

    def test_missing_required_quantity_is_rejected(self) -> None:
        case = _case()
        del case["physics"]["environment"]["spring_stiffness"]

        with self.assertRaisesRegex(
            ValueError,
            "required.*spring_stiffness|invalid.*environment",
        ):
            self.registry.render(case, case["text"]["prompt"])

    def test_ascii_symbol_requires_identifier_boundaries(self) -> None:
        case = _case()
        case["text"]["prompt"] = case["text"]["prompt"].replace(
            "mass m", "mass mm"
        )

        with self.assertRaisesRegex(ValueError, "symbol.*m.*absent"):
            self.registry.render(case, case["text"]["prompt"])


if __name__ == "__main__":
    unittest.main()
