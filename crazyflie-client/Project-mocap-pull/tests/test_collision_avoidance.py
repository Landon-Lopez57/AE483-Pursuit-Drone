"""Run with: python -m unittest discover -s tests -v"""
import unittest
import numpy as np

from collision_avoidance import (
    AvoidanceSettings, AvoidanceState, avoidance_step, closest_separation,
)


class AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.lower = np.array([-1.0, -1.0])
        self.upper = np.array([1.0, 1.0])

    def test_clear_path_keeps_tracking_command(self):
        command, status = avoidance_step(
            [0, 0], [0, 0], [0, 0], [0.04, 0], [],
            self.lower, self.upper, 0.05, AvoidanceState())
        np.testing.assert_allclose(command, [0.04, 0])
        self.assertEqual(status, 'tracking')

    def test_prediction_finds_crossing_and_stationary_distance(self):
        self.assertAlmostEqual(closest_separation(
            np.array([1., 0]), np.array([-1., 0]), 2), 0)
        self.assertAlmostEqual(closest_separation(
            np.array([1., 0]), np.zeros(2), 2), 1)

    def test_close_neighbor_stops(self):
        _, status = avoidance_step(
            [0, 0], [0, 0], [0, 0], [0.05, 0], [([0.05, 0], [0, 0])],
            self.lower, self.upper, 0.05, AvoidanceState())
        self.assertEqual(status, 'stop')

    def test_multiple_neighbors_and_box_limits(self):
        old = np.array([0.98, 0.98])
        neighbors = [([0.75, 0.98], [0.1, 0]), ([0.98, 0.7], [0, 0.1])]
        command, status = avoidance_step(
            old, [0, 0], old, [1.2, 1.2], neighbors,
            self.lower, self.upper, 0.05, AvoidanceState())
        self.assertIn(status, ('avoiding', 'stop'))
        self.assertTrue(np.all(command <= self.upper))
        self.assertTrue(np.all(command >= self.lower))
        self.assertLessEqual(np.linalg.norm(command - old), 0.05 + 1e-10)

    def test_hysteresis_retains_side_then_releases(self):
        state = AvoidanceState(active=True, right=np.array([0., -1.]))
        _, status = avoidance_step(
            [0, 0], [0, 0], [0, 0], [-0.04, 0], [([0.5, 0], [0, 0])],
            self.lower, self.upper, 0.05, state)
        self.assertEqual(status, 'avoiding')
        np.testing.assert_allclose(state.right, [0, -1])
        _, status = avoidance_step(
            [0, 0], [0, 0], [0, 0], [-0.04, 0], [([0.7, 0], [0, 0])],
            self.lower, self.upper, 0.05, state)
        self.assertEqual(status, 'tracking')
        self.assertFalse(state.active)

    def test_encounters_with_response_lag(self):
        encounters = {
            'head-on': ([[-0.8, 0], [0.8, 0]], [[0.8, 0], [-0.8, 0]]),
            'crossing': ([[-0.7, 0], [0, -0.7]], [[0.7, 0], [0, 0.7]]),
            'overtaking': ([[-0.8, 0], [-0.35, 0]], [[0.8, 0], [0.8, 0]]),
            'boundary': ([[-0.8, 0.9], [0.8, 0.9]], [[0.8, 0.9], [-0.8, 0.9]]),
        }
        for name, (start, goals) in encounters.items():
            with self.subTest(name=name):
                positions = np.array(start, dtype=float)
                commands = positions.copy()
                velocities = np.zeros((2, 2))
                states = [AvoidanceState(), AvoidanceState()]
                avoided = False
                for _ in range(160):
                    proposals, statuses = [], []
                    for i in range(2):
                        delta = np.array(goals[i]) - commands[i]
                        length = np.linalg.norm(delta)
                        desired = commands[i] + delta * min(1., 0.05 / max(length, 1e-9))
                        next_command, status = avoidance_step(
                            positions[i], velocities[i], commands[i], desired,
                            [(positions[1-i], velocities[1-i])], self.lower, self.upper,
                            0.05, states[i])
                        proposals.append(next_command)
                        statuses.append(status)
                        self.assertLessEqual(np.linalg.norm(next_command - commands[i]),
                                             0.05 + 1e-10)
                        self.assertTrue(np.all(next_command >= self.lower))
                        self.assertTrue(np.all(next_command <= self.upper))
                    avoided |= 'avoiding' in statuses
                    if 'stop' in statuses:
                        break
                    commands = np.array(proposals)
                    # The leader in the overtaking case responds more slowly.
                    tau = np.array([0.5, 1.0 if name == 'overtaking' else 0.5])
                    next_positions = positions + (1 - np.exp(-0.1 / tau))[:, None] * (
                        commands - positions)
                    velocities = (next_positions - positions) / 0.1
                    distance = closest_separation(
                        positions[1] - positions[0], velocities[1] - velocities[0], 0.1)
                    self.assertGreaterEqual(distance, 0.1 - 1e-10)
                    positions = next_positions
                self.assertTrue(avoided)


if __name__ == '__main__':
    unittest.main()
