"""Check conditional-label support, original weights and probability composition."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from factorized_contract import side_scores, targets, weights

from AstraResearch.io import ContractError


class FactorizedContractTest(unittest.TestCase):
    def test_conditional_population_uses_original_native_endpoint_indicators(self):
        rows = pd.DataFrame(
            {"mid_return_ticks[300s]": [-5.0, 0.0, 5.0, np.nan], "mid_endpoint.up.5[300s]": [0.0, 0.0, 1.0, np.nan], "mid_endpoint.down.5[300s]": [1.0, 0.0, 0.0, np.nan]}
        )
        codebook = targets(rows)
        np.testing.assert_array_equal(codebook["known"], [True, True, True, False])
        np.testing.assert_array_equal(codebook["big"], [True, False, True, False])
        np.testing.assert_array_equal(codebook["event_label"], [1, 0, 1])
        np.testing.assert_array_equal(codebook["direction_label"], [0, 1])

    def test_symbol_day_weights_are_formed_before_conditional_event_selection(self):
        rows = pd.DataFrame({"day": ["20260102"] * 4 + ["20260105"] * 2, "symbol": ["2330"] * 2 + ["2317"] * 2 + ["2330"] * 2})
        value = weights(rows, np.ones(6, dtype=bool), 40)
        self.assertAlmostEqual(value.mean(), 1)
        self.assertAlmostEqual(value[0], value[2])
        self.assertAlmostEqual(value[4] / value[0], 2 ** (3 / 40))
        # Conditioning one event in one group and two in another preserves
        # their original relative row mass; it does not rebalance event groups.
        conditional = value[[0, 4, 5]]
        self.assertAlmostEqual(conditional[1] / conditional[0], value[4] / value[0])

    def test_composed_side_probabilities_have_coherent_total_event_mass(self):
        value = side_scores([0, 0.2, 1], [0.4, 0.75, 0])
        np.testing.assert_allclose(value, [[0, 0], [0.15, 0.05], [0, 1]])
        np.testing.assert_allclose(value.sum(axis=1), [0, 0.2, 1])
        for event, direction in [([np.nan], [0.5]), ([1.01], [0.5]), ([0.5], [-0.01]), ([0.5, 0.2], [0.5])]:
            with self.assertRaises(ContractError):
                side_scores(event, direction)


if __name__ == "__main__":
    unittest.main()
