from __future__ import annotations

import unittest
from pathlib import Path

from _paths import ROOT


class ReferenceObservationCurationIdentityTests(unittest.TestCase):
    @staticmethod
    def _identity_api():
        from physbench.reference_observations.curation import identity

        return identity

    def test_collision_visual_order_rejects_reversed_physics_identity(self) -> None:
        identity = self._identity_api()
        entities = [
            identity.EntityIdentity(
                object_id="object_1",
                mask_id="01",
                centroid_xy=(812.0, 301.0),
                bbox_xyxy=(799.0, 288.0, 825.0, 314.0),
            ),
            identity.EntityIdentity(
                object_id="object_2",
                mask_id="02",
                centroid_xy=(774.0, 304.0),
                bbox_xyxy=(764.0, 294.0, 784.0, 314.0),
            ),
        ]

        issues = identity.audit_entity_order("collision_1d", entities)

        self.assertEqual(
            ["collision_left_to_right_order"],
            [issue.code for issue in issues],
        )
        self.assertEqual(("object_2", "object_1"), issues[0].observed_order)
        self.assertEqual(("object_1", "object_2"), issues[0].expected_order)

    def test_collision_visual_order_accepts_left_to_right_physics_identity(self) -> None:
        identity = self._identity_api()
        entities = [
            identity.EntityIdentity(
                object_id="object_1",
                mask_id="01",
                centroid_xy=(100.0, 302.0),
                bbox_xyxy=(88.0, 290.0, 112.0, 314.0),
            ),
            identity.EntityIdentity(
                object_id="object_2",
                mask_id="02",
                centroid_xy=(180.0, 300.0),
                bbox_xyxy=(170.0, 290.0, 190.0, 310.0),
            ),
        ]

        self.assertEqual((), identity.audit_entity_order("collision_1d", entities))

    def test_circular_order_groups_same_row_before_sorting_left_to_right(self) -> None:
        identity = self._identity_api()
        entities = [
            identity.EntityIdentity(
                object_id="object_1",
                mask_id="01",
                centroid_xy=(150.0, 100.0),
                bbox_xyxy=(140.0, 90.0, 160.0, 110.0),
            ),
            identity.EntityIdentity(
                object_id="object_2",
                mask_id="02",
                centroid_xy=(90.0, 108.0),
                bbox_xyxy=(80.0, 98.0, 100.0, 118.0),
            ),
            identity.EntityIdentity(
                object_id="object_3",
                mask_id="03",
                centroid_xy=(40.0, 180.0),
                bbox_xyxy=(30.0, 170.0, 50.0, 190.0),
            ),
        ]

        issues = identity.audit_entity_order("uniform_circular_motion", entities)

        self.assertEqual(["row_major_object_order"], [issue.code for issue in issues])
        self.assertEqual(
            ("object_2", "object_1", "object_3"),
            issues[0].observed_order,
        )

    def test_mask_physics_keys_must_exactly_cover_object_quantities(self) -> None:
        identity = self._identity_api()

        issues = identity.audit_physics_key_binding(
            object_id="object_1",
            quantity_names={"mass", "radius", "initial_velocity"},
            physics_keys=[
                "objects.object_1.mass",
                "objects.object_1.radius",
            ],
        )

        self.assertEqual(["mask_physics_keys_mismatch"], [issue.code for issue in issues])
        self.assertEqual(
            ("objects.object_1.initial_velocity",),
            issues[0].missing_keys,
        )
        self.assertEqual((), issues[0].unexpected_keys)

    def test_mask_physics_keys_reject_cross_object_binding(self) -> None:
        identity = self._identity_api()

        issues = identity.audit_physics_key_binding(
            object_id="object_1",
            quantity_names={"mass"},
            physics_keys=["objects.object_2.mass"],
        )

        self.assertEqual(
            ("objects.object_1.mass",),
            issues[0].missing_keys,
        )
        self.assertEqual(
            ("objects.object_2.mass",),
            issues[0].unexpected_keys,
        )


class ReferenceObservationCurationCatalogTests(unittest.TestCase):
    DATASET = ROOT / "datasets" / "releases" / "14.0.0" / "dataset.json"
    CIRCULAR_CASE_ID = "circular_r1_silver02cm_img_0370"

    @staticmethod
    def _curation_api():
        import physbench.reference_observations.curation as curation

        return curation

    def test_catalog_loads_case_local_members_into_one_binding(self) -> None:
        curation = self._curation_api()

        cases = curation.load_curation_cases(
            self.DATASET,
            case_ids={self.CIRCULAR_CASE_ID},
        )

        self.assertEqual(1, len(cases))
        case = cases[0]
        self.assertEqual(self.CIRCULAR_CASE_ID, case.case_id)
        self.assertEqual("uniform_circular_motion", case.scene_id)
        self.assertEqual(
            "An object follows a circular orbit of radius r_1 with angular speed ω.",
            case.caption,
        )
        self.assertEqual(("object_1",), tuple(case.physics["objects"]))
        self.assertEqual(1, len(case.entities))
        entity = case.entities[0]
        self.assertEqual("object_1", entity.identity.object_id)
        self.assertEqual("01", entity.identity.mask_id)
        self.assertEqual(
            ("objects.object_1.orbit_radius",),
            entity.physics_keys,
        )
        for path in (
            case.first_frame_path,
            case.reference_video_path,
            entity.anchor_npz_path,
            entity.mask_tube_path,
            entity.trajectory_path,
        ):
            self.assertTrue(Path(path).is_file(), path)
        self.assertEqual(356, len(case.timeline["samples"]))
        self.assertEqual((), curation.audit_case_bindings(case))

    def test_catalog_rejects_requested_case_id_missing_from_release(self) -> None:
        curation = self._curation_api()

        with self.assertRaisesRegex(ValueError, "unknown requested Case IDs"):
            curation.load_curation_cases(
                self.DATASET,
                case_ids={"not_a_real_case"},
            )


if __name__ == "__main__":
    unittest.main()
