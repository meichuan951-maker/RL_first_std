"""从训练 CSV 绘制关闭探索后的评估胜率、平局率和失败率。"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--log", type=Path, default=project_dir / "results" / "training.csv")
    parser.add_argument("--output", type=Path, default=project_dir / "results" / "learning_curve.png")
    args = parser.parse_args()
    try:
        with args.log.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError("训练记录为空")
        episodes = [int(row["episode"]) for row in rows]
        rates = {
            name: [float(row[f"eval_{name}_rate"]) for row in rows]
            for name in ("win", "draw", "loss")
        }
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"读取训练记录失败：{exc}。请先运行 train.py。\n")

    try:
        import matplotlib
        matplotlib.use("Agg")  # 没有桌面的机器也可以保存图片。
        import matplotlib.pyplot as plt
    except ImportError:
        parser.exit(1, "绘图需要 matplotlib：python3 -m pip install matplotlib\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in (("win", "#198754"), ("draw", "#3079c9"), ("loss", "#dc3545")):
        ax.plot(episodes, rates[name], label=name.capitalize(), color=color, linewidth=2)
    ax.set(xlabel="Training episodes", ylabel="Rate", ylim=(0, 1.02),
           title="Greedy policy vs random opponent (both sides)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"学习曲线已保存：{args.output}")


if __name__ == "__main__":
    main()
