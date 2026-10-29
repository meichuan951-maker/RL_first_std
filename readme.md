# 井字棋强化学习：运行与核心函数

这个项目用 Q-learning 学习下棋。环境负责执行落子和判定输赢，智能体根据 Q 表选动作，并用对局经验更新 Q 表。Q 表可以理解为“某个棋盘下，每个落子位置的评分”。

**当前状态：`environment.py` 和 `agent.py` 是包含 `pass` 的练习骨架，部分接口还与调用方不一致。训练、评估和对战需要补全第 4 节列出的接口后才能运行；已有模型文件也不能代替这些代码。**

## 1. 文件分工

```text
tic_tac_toe_rl/
├── environment.py    # 棋盘、合法动作、输赢和奖励
├── agent.py          # 选动作、更新 Q 表、保存和读取模型
├── train.py          # 组织训练流程，写入模型和训练日志
├── evaluate.py       # 加载模型，统计先后手的胜、平、负
├── play.py           # 在终端与 AI 对战
├── plot.py           # 读取训练日志，保存学习曲线
├── models/           # 默认模型：q_table.json
└── results/          # 默认日志：training.csv；图片：learning_curve.png
```

## 2. 怎么运行

使用 Python 3.10 或更新版本。补全骨架后，在终端依次执行：

```bash
cd /home/user/RL_stduy/tic_tac_toe_rl

# 训练：与随机对手下 10 万局，AI 交替执先手、后手
python3 train.py --episodes 100000

# 评估：对随机对手、minimax 最优对手，各测试 2000 局
python3 evaluate.py --games 2000

# 对战：你先手；去掉 --human-first 则 AI 先手
python3 play.py --human-first
```

训练结束后会保存 `models/q_table.json` 和 `results/training.csv`；评估和对战读取模型，不更新 Q 表。训练日志中，`ε` 是随机探索概率，`状态数` 是 Q 表记录的棋盘数，胜平负是在关闭探索后评估得到的比例。

对战时输入空格编号 `0～8` 落子，输入 `q` 退出；一局结束后输入 `y` 再来一局，并交换先后手。

```text
0 | 1 | 2
--+---+--
3 | 4 | 5
--+---+--
6 | 7 | 8
```

可选绘图，只有这一步需要 matplotlib：

```bash
python3 -m pip install matplotlib
python3 plot.py
```

图片保存到 `results/learning_curve.png`。默认输出路径相对于脚本目录；再次训练会覆盖默认模型和日志，可用 `--model`、`--log` 指定其他路径。

## 3. 核心函数做什么

### 环境：`environment.py`

- `reset()`：开始新一局，返回初始状态；AI 后手时，先让对手落一子。
- `get_valid_actions()`：返回可以落子的空格编号。**当前调用方要求这个方法叫 `legal_actions()`，需要统一名称。**
- `check_winner()`：检查棋盘是否有人获胜或已经平局。
- `step(action)`：执行 AI 的动作；若未结束，再让对手回应，返回 `(next_state, reward, done, info)`。
- `render()`：在终端打印棋盘。

`state` 是长度为 9 的元组：`0` 表示空格，`1` 表示自己的棋子，`-1` 表示对方的棋子。`action` 是落子编号。`reward` 按 AI 的结果计算：赢 `+1`、输 `-1`、平局或未结束 `0`。`done=True` 表示本局结束，`info` 保存获胜方等附加信息。

### 智能体：`agent.py`

- `__init__(alpha, gamma, epsilon)`：创建 Q 表并设置参数；`alpha` 控制更新幅度，`gamma` 控制未来收益折扣，`epsilon` 控制随机探索概率。
- `choose_action(state, valid_actions)`：训练时以 ε 的概率随机探索，否则选择合法动作中 Q 值最高的动作。
- `update(state, action, reward, next_state, next_valid_actions, done)`：根据这一步经验更新一个 Q 值；终局只看奖励，非终局还考虑下一状态最好的合法动作价值。
- `save(path)`：把参数和 Q 表保存成 JSON 文件。
- `load(path)`：读取 JSON 模型，返回可用于评估或对战的智能体。

### 程序入口

- `train.py` 的 `train()`：创建环境和智能体，反复选动作、执行动作、更新 Q 表，定期评估并在结束时保存模型。
- `evaluate.py` 的 `evaluate()`：关闭探索、交替先后手对战，返回总计及先手、后手的胜平负；`minimax_opponent()` 提供最优对手用于检验策略。
- `play.py` 的 `human_opponent()`：读取并检查真人输入，作为环境中的对手策略。
- 各脚本的 `main()`：读取命令行参数，启动对应流程；`plot.py` 的 `main()` 负责读取 CSV 和保存图片。

## 4. 接通现有脚本前，需要补齐的接口

以下是调用方已经使用、骨架尚未提供的接口：

- **环境构造与重置：**支持 `TicTacToeEnv(opponent=None, seed=None)` 和 `reset(agent_player=1)`；`agent_player=1` 表示 AI 执 X 先手，`-1` 表示执 O 后手。
- **环境方法：**提供 `env.legal_actions()`；`step()` 按第 3 节返回四个值。一次 `step()` 结束后，若棋局未结束，应再次轮到 AI。
- **环境模块函数：**提供 `get_winner(board)`、`legal_actions(board)`、`random_opponent(board, player, rng)`，供评估脚本导入。这里原始棋盘用 `1` 表示 X、`-1` 表示 O；`get_winner()` 返回 `1`（X 胜）、`-1`（O 胜）、`0`（平局）或 `None`（未结束）。类中的 `check_winner()` 不能直接替代模块级函数。
- **智能体初始化：**支持 `seed` 参数，并创建 `q_table`、`epsilon`、`rng`（`random.Random` 对象），供训练和评估使用。
- **智能体选择与加载：**`choose_action()` 支持 `explore=False`，供评估和对战关闭探索；`load()` 需要是类方法，支持 `QLearningAgent.load(path, seed=42)` 并返回智能体实例。

当前直接运行 `train.py`，会在导入阶段因缺少模块级 `get_winner()` 报错；应先补齐并统一接口。

## 5. 一局训练的核心流程

下面展示接口补齐后的调用顺序；`env` 和 `agent` 分别是环境、智能体实例：

```python
state = env.reset(agent_player=1)
done = False
while not done:
    action = agent.choose_action(state, env.legal_actions())
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, env.legal_actions(), done)
    state = next_state
```

先按这个顺序读代码：`train()` → 环境的 `reset()` / `step()` → 智能体的 `choose_action()` / `update()`。其中真正改变 AI 经验的是 `update()`。
