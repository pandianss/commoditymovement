import unittest
from core.risk import RiskProfile, CapitalConstitution, PortfolioState

class TestRiskConstitution(unittest.TestCase):
    def setUp(self):
        self.profile = RiskProfile(
            max_drawdown=0.20,
            vol_target=0.15,
            max_leverage=2.0,
            position_limit_pct=0.50
        )
        self.constitution = CapitalConstitution(self.profile)

    def test_drawdown_check(self):
        # Good State
        state = PortfolioState(
            current_equity=95000,
            peak_equity=100000, # 5% DD
            current_volatility=0.10,
            gross_exposure=0,
            positions={}
        )
        self.assertTrue(self.constitution.check_drawdown(state))
        
        # Bad State
        state_bad = PortfolioState(
            current_equity=70000,
            peak_equity=100000, # 30% DD > 20% limit
            current_volatility=0.10,
            gross_exposure=0,
            positions={}
        )
        self.assertFalse(self.constitution.check_drawdown(state_bad))

    def test_vol_scalar(self):
        # Low Vol
        state = PortfolioState(
            current_equity=100000,
            peak_equity=100000,
            current_volatility=0.10, # < 0.15 target
            gross_exposure=0,
            positions={}
        )
        self.assertAlmostEqual(self.constitution.get_vol_scalar(state), 1.0)
        
        # High Vol
        state_high = PortfolioState(
            current_equity=100000,
            peak_equity=100000,
            current_volatility=0.30, # Double target
            gross_exposure=0,
            positions={}
        )
        self.assertAlmostEqual(self.constitution.get_vol_scalar(state_high), 0.5)

    def test_position_clamping(self):
        state = PortfolioState(
            current_equity=100000,
            peak_equity=100000,
            current_volatility=0.15,
            gross_exposure=0,
            positions={}
        )
        # Limit is 50% -> 50,000
        # Request 80,000
        approved = self.constitution.validate_position_size("TEST", 80000, state)
        self.assertEqual(approved, 50000)
        
        # Request -60,000
        approved_short = self.constitution.validate_position_size("TEST", -60000, state)
        self.assertEqual(approved_short, -50000)

if __name__ == '__main__':
    unittest.main()
