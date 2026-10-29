"""与随机对手下棋，用表格型 Q-learning 学习每个状态的落子价值。"""

import argparse
import csv
from pathlib import Path

if __package__:
    from .agent import QLearningAgent
    from .environment import TicTacToeEnv
    from .evaluate import evaluate
else:
    from agent import QLearningAgent
    from environment import TicTacToeEnv
    from evaluate import evaluate


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "q_table.json"
DEFAULT_LOG = PROJECT_DIR / "results" / "training.csv"


def train(
    episodes=100_000,
    alpha=0.2,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=42,
    log_every=5_000,
    eval_games=500,
    model_path=DEFAULT_MODEL,
    log_path=DEFAULT_LOG,
):
    """一局棋是一条 episode；先后手交替，定期关闭探索做评估。"""
    if episodes <= 0 or log_every <= 0 or eval_games <= 0:
        raise ValueError("episodes、log_every 和 eval_games 必须是正整数")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("需要满足 0 <= epsilon_end <= epsilon_start <= 1")

    env = TicTacToeEnv(seed=seed)
    agent = QLearningAgent(alpha, gamma, epsilon_start, seed=seed)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode", "epsilon", "states",
        "train_win_rate", "train_draw_rate", "train_loss_rate",
        "eval_win_rate", "eval_draw_rate", "eval_loss_rate",
    ]
    # 窗口内的训练成绩包含随机探索；评估成绩才反映当前贪心策略的表现。
    counts = {"wins": 0, "draws": 0, "losses": 0}
    window_games = 0

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        def record(episode):
            result = evaluate(agent, games=eval_games, seed=seed + 1)
            row = {
                "episode": episode,
                "epsilon": agent.epsilon,
                "states": len(agent.q_table),
                "train_win_rate": counts["wins"] / window_games if window_games else "",
                "train_draw_rate": counts["draws"] / window_games if window_games else "",
                "train_loss_rate": counts["losses"] / window_games if window_games else "",
                "eval_win_rate": result["win_rate"],
                "eval_draw_rate": result["draw_rate"],
                "eval_loss_rate": result["loss_rate"],
            }
            writer.writerow(row)
            file.flush()
            print(
                f"局数 {episode:>6}/{episodes} | ε={agent.epsilon:.3f} "
                f"| 状态数 {len(agent.q_table):>4} | 对随机对手评估："
                f"胜 {result['win_rate']:.1%} / 平 {result['draw_rate']:.1%} "
                f"/ 负 {result['loss_rate']:.1%}"
            )

        record(0)  # 训练前的基线，便于判断是否真的学到了东西。
        for episode in range(1, episodes + 1):
            # 前 80% 的训练逐渐减少探索，最后 20% 保持少量探索。
            progress = min((episode - 1) / max(1, episodes * 0.8 - 1), 1.0)
            agent.epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
            state = env.reset(agent_player=1 if episode % 2 else -1)
            done = False

            while not done:
                action = agent.choose_action(state, env.legal_actions())
                next_state, reward, done, _ = env.step(action)
                agent.update(
                    state, action, reward, next_state, env.legal_actions(), done
                )
                state = next_state

            key = "wins" if reward > 0 else "losses" if reward < 0 else "draws"
            counts[key] += 1
            window_games += 1
            if episode % log_every == 0 or episode == episodes:
                record(episode)
                counts = {"wins": 0, "draws": 0, "losses": 0}
                window_games = 0

    agent.save(model_path)
    print(f"模型已保存：{model_path}")
    print(f"训练记录：{log_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000, help="训练局数")
    parser.add_argument("--alpha", type=float, default=0.2, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    try:
        train(
            episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            seed=args.seed, log_every=args.log_every, eval_games=args.eval_games,
            model_path=args.model, log_path=args.log,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"训练失败：{exc}\n")


if __name__ == "__main__":
    main()
