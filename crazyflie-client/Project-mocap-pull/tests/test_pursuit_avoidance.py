import unittest
import numpy as np
from match_config import CONFIG
from pursuit_avoidance import PursuitAvoidance
from match_simulation import simulate_match, validate_match, scoring_mask


class MatchTests(unittest.TestCase):
    def test_full_match_respects_constraints(self):
        result = simulate_match()
        validate_match(result)
        self.assertTrue(np.any(result['statuses'] == 'avoiding'))
        self.assertGreater(np.ptp(result['positions'][:, 0, 0]), 0.5)

    def test_rear_start_completes_and_accumulates_score(self):
        result = simulate_match(starts=[[-0.5,0,0.5],[0,0,0.5]],
                                initial_headings=[[1,0],[1,0]])
        validate_match(result)
        self.assertGreater(result['final_scores'][0], 0)
        self.assertEqual(result['outcome'], 'Drone 1 wins')

    def test_scoring_rear_distance_and_angle(self):
        headings = np.array([[1., 0], [1., 0]])
        for x, y, expected in [(-0.5, 0, True), (-0.2, 0, False),
                               (-0.7, 0, False), (0.5, 0, False), (-0.5, 0.2, False)]:
            p = np.array([[x, y, CONFIG.height], [0, 0, CONFIG.height]])
            self.assertEqual(bool(scoring_mask(p, headings)[0]), expected)
        score = sum(scoring_mask(np.array([[-0.5,0,0.5],[0,0,0.5]]), headings)[0]
                    * CONFIG.period for _ in range(round(10 / CONFIG.period)))
        self.assertAlmostEqual(score, 10.)

    def test_missing_velocity_stops(self):
        c = PursuitAvoidance([-1, 0, CONFIG.height])
        command, _, status = c.step(c.sp, None, [1,0,0.5], [0,0,0], [1,0])
        self.assertEqual(status, 'stop')
        np.testing.assert_array_equal(command, c.sp)

    def test_measured_overlap_stops(self):
        c = PursuitAvoidance([0,0,CONFIG.height])
        _, _, status = c.step(c.sp, [0,0,0], [0.05,0,0.5], [0,0,0], [1,0])
        self.assertEqual(status, 'stop')
