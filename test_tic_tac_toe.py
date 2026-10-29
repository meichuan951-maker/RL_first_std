"""运行：python3 -m unittest discover -s tic_tac_toe_rl -v"""

import random
import tempfile
import unittest
from pathlib import Path

if __package__:
    from .agent import QLearningAgent
    from .environment import TicTacToeEnv, get_winner, legal_actions
    from .evaluate import evaluate, minimax_opponent
else:
    from agent import QLearningAgent
    from environment import TicTacToeEnv, get_winner, legal_actions
    from evaluate import evaluate, minimax_opponent


def scripted_opponent(moves):
    moves = iter(moves)
    return lambda board, player, rng: next(moves)


class EnvironmentTests(unittest.TestCase):
    def test_agent_win_stops_before_opponent_reply(self):
        env = TicTacToeEnv(opponent=scripted_opponent([3, 4]))
        env.reset()
        env.step(0)
        env.step(1)
        _, reward, done, info = env.step(2)
        self.assertEqual((reward, done, info["winner"]), (1, True, 1))
        self.assertEqual(env.legal_actions(), [])
        with self.assertRaises(ValueError):
            env.step(8)

    def test_opponent_win_has_negative_reward(self):
        env = TicTacToeEnv(opponent=scripted_opponent([3, 4, 5]))
        env.reset()
        env.step(0)
        env.step(8)
        _, reward, done, _ = env.step(7)
        self.assertEqual((reward, done), (-1, True))

    def test_draw_has_zero_reward(self):
        env = TicTacToeEnv(opponent=scripted_opponent([1, 4, 5, 6]))
        env.reset()
        for action in (0, 2, 3, 7, 8):
            _, reward, done, info = env.step(action)
        self.assertEqual((reward, done, info["winner"]), (0, True, 0))

    def test_second_player_state_uses_own_perspective(self):
        env = TicTacToeEnv(opponent=scripted_opponent([4, 8]))
        state = env.reset(agent_player=-1)
        self.assertEqual(state[4], -1)
        state, _, done, _ = env.step(0)
        self.assertFalse(done)
        self.assertEqual((state[0], state[4], state[8]), (1, -1, -1))

    def test_invalid_action_does_not_change_board(self):
        env = TicTacToeEnv(opponent=scripted_opponent([4]))
        env.reset()
        env.step(0)
        before = env.board.copy()
        for action in (-1, 9, 0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.board, before)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.state = (0,) * 9
        self.next_state = (1, -1, 0, 0, 0, 0, 0, 0, 0)
        self.agent = QLearningAgent(alpha=0.2, gamma=0.9, seed=42)

    def test_bootstrap_ignores_occupied_cells(self):
        self.agent.q_table[self.next_state] = [100.0, 100.0, 0.5] + [0.0] * 6
        self.agent.update(self.state, 0, 0, self.next_state, list(range(2, 9)), False)
        self.assertAlmostEqual(self.agent.q_table[self.state][0], 0.09)

    def test_terminal_update_does_not_bootstrap(self):
        self.agent.q_table[self.next_state] = [100.0] * 9
        self.agent.update(self.state, 0, -1, self.next_state, [], True)
        self.assertAlmostEqual(self.agent.q_table[self.state][0], -0.2)

    def test_greedy_choice_ignores_occupied_cells(self):
        self.agent.q_table[self.next_state] = [100.0, 100.0, 0.5] + [0.0] * 6
        self.assertEqual(self.agent.choose_action(self.next_state, range(2, 9), False), 2)

    def test_json_round_trip(self):
        self.agent.update(self.state, 4, 1, self.next_state, [], True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            self.agent.save(path)
            restored = QLearningAgent.load(path)
            self.assertEqual(restored.q_table, self.agent.q_table)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                QLearningAgent.load(path)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_is_reproducible_and_preserves_training_state(self):
        agent = QLearningAgent(seed=7)
        rng_state = agent.rng.getstate()
        result = evaluate(agent, games=101, seed=42)
        self.assertEqual(agent.rng.getstate(), rng_state)
        self.assertEqual(agent.q_table, {})
        self.assertEqual(agent.epsilon, 1)
        self.assertEqual(result, evaluate(agent, games=101, seed=42))
        self.assertEqual((result["first"]["games"], result["second"]["games"]), (51, 50))
        self.assertEqual(sum(result[key] for key in ("wins", "draws", "losses")), 101)

    def test_minimax_cannot_lose_against_any_reply(self):
        # 穷举对手的所有应对，而不只抽样若干局，检验评估基准确实不会输。
        for ai_side in (1, -1):
            rng = random.Random(42)

            def visit(board, player):
                winner = get_winner(board)
                if winner is not None:
                    self.assertIn(winner, (0, ai_side))
                    return
                actions = ([minimax_opponent(board, player, rng)] if player == ai_side
                           else legal_actions(board))
                for action in actions:
                    next_board = list(board)
                    next_board[action] = player
                    visit(tuple(next_board), -player)

            visit((0,) * 9, 1)


if __name__ == "__main__":
    unittest.main()
