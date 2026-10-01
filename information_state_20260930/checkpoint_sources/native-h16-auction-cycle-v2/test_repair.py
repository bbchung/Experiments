"""No native data: rejected binding repair preserves all scientific identities."""

import unittest

import repair_pipeline as repair


class BindingRepairTest(unittest.TestCase):
    def fixture(self):
        profile = {"path": "/frozen/profile.yaml", "sha256": "a" * 64, "size": 10}
        core = {"path": "/frozen/cpp", "sha256": "b" * 64, "size": 20}
        parent = {
            "identity": repair.PARENT_ID,
            "status": "frozen",
            "schema": repair.study.PRODUCER_SCHEMA,
            "sources": [profile, core],
            "inputs": [core],
            "preparation_identity": "c" * 64,
            "profile_identity": "d" * 64,
        }
        return parent, profile

    def test_only_missing_same_profile_binding_is_extended(self):
        parent, profile = self.fixture()
        result = repair.repaired_payload(parent, profile, [], [])
        self.assertEqual(result["sources"], sorted(parent["sources"], key=lambda node: node["path"]))
        self.assertEqual(result["inputs"], sorted([*parent["inputs"], profile], key=lambda node: node["path"]))
        self.assertEqual(result["preparation_identity"], parent["preparation_identity"])
        self.assertEqual(result["profile_identity"], parent["profile_identity"])
        self.assertFalse(result["feature_math_or_procedure_changed"])
        self.assertEqual(parent["inputs"], [parent["sources"][1]])
        self.assertNotIn("identity", result)

    def test_different_profile_or_already_bound_parent_cannot_be_repaired(self):
        parent, profile = self.fixture()
        for changed in ({**profile, "sha256": "e" * 64}, {**profile, "path": "/other/profile.yaml"}):
            with self.subTest(profile=changed), self.assertRaises(ValueError):
                repair.repaired_payload(parent, changed, [], [])
        with self.assertRaisesRegex(ValueError, "missing-input"):
            repair.repaired_payload({**parent, "inputs": [*parent["inputs"], profile]}, profile, [], [])
        with self.assertRaisesRegex(ValueError, "wrong rejected"):
            repair.repaired_payload({**parent, "identity": "0" * 64}, profile, [], [])


if __name__ == "__main__":
    unittest.main()
