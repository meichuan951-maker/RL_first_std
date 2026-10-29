"""在终端里与训练好的 AI 对弈，输入 0～8 选择格子。"""

import argparse
from pathlib import Path

if __package__:
    from .agent import QLearningAgent
    from .environment import TicTacToeEnv
else:
    from agent import QLearningAgent
    from environment import TicTacToeEnv


def human_opponent(board, player, rng):
    """环境调用这个函数，让真人完成对手的一步。"""
    del rng
    symbols = {1: "X", -1: "O"}
    print(f"\n轮到你（{symbols[player]}）：")
    for row in range(3):
        print(" | ".join(symbols.get(board[i], str(i)) for i in range(row * 3, row * 3 + 3)))
        if row < 2:
            print("--+---+--")
    while True:
        value = input("选择空格 0～8（q 退出）：").strip().lower()
        if value == "q":
            raise EOFError
        try:
            action = int(value)
        except ValueError:
            print("请输入 0～8 的整数。")
            continue
        if 0 <= action < 9 and board[action] == 0:
            return action
        print("这个位置不能落子，请选择一个空格。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "q_table.json",
    )
    parser.add_argument("--human-first", action="store_true", help="你执 X 先手；默认 AI 先手")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        agent = QLearningAgent.load(args.model, seed=args.seed)
    except FileNotFoundError:
        parser.exit(1, "找不到模型，请先运行 python3 train.py。\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"读取模型失败：{exc}\n")

    env = TicTacToeEnv(opponent=human_opponent, seed=args.seed)
    ai_player = -1 if args.human_first else 1
    print("你执 X，AI 执 O。" if args.human_first else "AI 执 X，你执 O。")
    try:
        while True:
            state = env.reset(agent_player=ai_player)
            done = False
            while not done:
                action = agent.choose_action(state, env.legal_actions(), explore=False)
                print(f"\nAI 落在 {action}。")
                state, reward, done, _ = env.step(action)
            print("\n最终棋盘：")
            env.render()
            print("AI 获胜。" if reward > 0 else "你获胜了！" if reward < 0 else "平局。")
            if input("再来一局？[y/N] ").strip().lower() != "y":
                break
            ai_player = -ai_player  # 下一局交换先后手。
    except (EOFError, KeyboardInterrupt):
        print("\n已退出对战。")


if __name__ == "__main__":
    main()
