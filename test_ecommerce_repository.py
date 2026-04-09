#!/usr/bin/env python3
"""Regression tests for ecommerce repository field normalization."""

import unittest

from skills.ecommerce_repository import load_customers, normalize_column_name


class EcommerceRepositoryTests(unittest.TestCase):
    def test_prime_subscription_header_alias_maps_to_expected_key(self) -> None:
        self.assertEqual(
            normalize_column_name("prime subscription flag (1/0)"),
            "prime_subscription_flag",
        )

    def test_prime_subscription_flag_loads_as_boolean(self) -> None:
        customers = load_customers()
        self.assertTrue(customers["1"]["prime_subscription_flag"])
        self.assertFalse(customers["2"]["prime_subscription_flag"])


if __name__ == "__main__":
    unittest.main()
