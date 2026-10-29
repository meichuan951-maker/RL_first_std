"""关闭探索，分别检查 AI 对随机对手和最优对手的表现。"""

import argparse
from functools import lru_cache
from pathlib import Path

if __package__:
    from .agent import QLearningAgent
    from .environment import TicTacToeEnv, get_winner, legal_actions, random_opponent
else:
    from agent import QLearningAgent
    from environment import TicTacToeEnv, get_winner, legal_actions, random_opponent


@lru_cache(maxsize=None)
def _minimax_value(board, player):
    """返回当前落子方在双方都最优时的结果：赢 1，平 0，输 -1。"""
    winner = get_winner(board)
    if winner is not None:
        return winner * player

    # 轮到对手后，评价的视角反过来，所以递归结果取负号。
    best = -1
    for action in legal_actions(board):
        next_board = list(board)
        next_board[action] = player
        best = max(best, -_minimax_value(tuple(next_board), -player))
        if best == 1:
            break
    return best


def minimax_opponent(board, player, rng):
    """用于检验策略的最优对手；它不参与 Q-learning 的更新。"""
    if player not in (1, -1):
        raise ValueError("player 必须是 1 或 -1")
    if get_winner(board) is not None:
        raise ValueError("棋局已经结束，不能再选择动作")

    best_value = -2
    best_actions = []
    for action in legal_actions(board):
        next_board = list(board)
        next_board[action] = player
        value = -_minimax_value(tuple(next_board), -player)
        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)
    return rng.choice(best_actions)


def _empty_counts():
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0}


def _add_rates(counts):
    games = counts["games"]
    for result in ("win", "draw", "loss"):
        key = {"win": "wins", "draw": "draws", "loss": "losses"}[result]
        counts[result + "_rate"] = counts[key] / games if games else 0.0
    return counts


def evaluate(agent, games=1000, seed=42, opponent="random"):
    """先后手交替评估，返回总计以及 first / second 的胜、平、负统计。

    比例介于 0 和 1；评估既不更新 Q 表，也不改变训练使用的随机数状态。
    奇数局时，先手比后手多一局。
    """
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("games 必须是正整数")
    opponents = {"random": random_opponent, "minimax": minimax_opponent}
    if opponent not in opponents:
        raise ValueError("opponent 必须是 random 或 minimax")

    env = TicTacToeEnv(opponent=opponents[opponent], seed=seed)
    first = _empty_counts()
    second = _empty_counts()
    # 并列的最优 Q 值仍需要随机选动作；单独播种以便复现评估结果。
    original_rng_state = agent.rng.getstate()
    agent.rng.seed(seed)
    try:
        for game in range(games):
            side = 1 if game % 2 == 0 else -1
            state = env.reset(agent_player=side)
            done = False
            reward = 0.0
            while not done:
                action = agent.choose_action(state, env.legal_actions(), explore=False)
                state, reward, done, _ = env.step(action)

            counts = first if side == 1 else second
            counts["games"] += 1
            result = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
            counts[result] += 1
    finally:
        agent.rng.setstate(original_rng_state)

    total = {key: first[key] + second[key] for key in first}
    _add_rates(total)
    total["first"] = _add_rates(first)
    total["second"] = _add_rates(second)
    return total


def print_report(result, opponent="random"):
    name = "随机对手" if opponent == "random" else "最优对手（minimax）"
    print(f"\n对战{name}，共 {result['games']} 局（关闭随机探索）")
    for label, counts in (("总计", result), ("先手", result["first"]), ("后手", result["second"])):
        print(
            f"  {label}：胜 {counts['wins']} / 平 {counts['draws']} / 负 {counts['losses']}"
            f"，胜率 {counts['win_rate']:.1%}，平局率 {counts['draw_rate']:.1%}"
            f"，败率 {counts['loss_rate']:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description="评估训练好的井字棋 Q-learning 策略")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models" / "q_table.json")
    parser.add_argument("--games", type=int, default=2000, help="每种对手的评估局数")
    parser.add_argument("--opponent", choices=("random", "minimax", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games 必须是正整数")
    try:
        agent = QLearningAgent.load(args.model, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.error(f"无法加载模型：{exc}。请先运行 train.py。")

    opponents = ("random", "minimax") if args.opponent == "both" else (args.opponent,)
    for opponent in opponents:
        print_report(evaluate(agent, args.games, args.seed, opponent), opponent)
    if "minimax" in opponents:
        print("\n井字棋双方都最优时会平局；对最优对手应重点看败率是否降低。")


if __name__ == "__main__":
    main()
