# MCTS.py
import math
import numpy as np

class MCTS():
    """
    一个支持批量预测的蒙特卡洛树搜索实现。
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # 存储Q值 (s,a)
        self.Nsa = {}  # 存储访问次数 (s,a)
        self.Ns = {}   # 存储状态s的访问次数
        self.Ps = {}   # 存储神经网络的初始策略P(s)

        self.Es = {}   # 存储游戏结束状态的返回值
        
        # 用于批量处理的新增成员
        self.nodes_to_evaluate = [] # 存储待评估的叶节点 (canonicalBoard, add_noise_flag)
        self.paths_to_backprop = [] # 存储到达每个待评估叶节点的路径

    def getActionProb(self, canonicalBoard, temp=1, add_noise=False):
        """
        执行MCTS模拟并返回策略向量。
        这个版本通过批量处理叶节点评估来优化性能。
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Ps:
            # 首次遇到该局面，先执行一次搜索以扩展根节点
            self.search(canonicalBoard, add_noise=add_noise)
            if self.nodes_to_evaluate:
                self._evaluate_and_backpropagate()

        for i in range(self.args.num_mcts_sims):
            self.search(canonicalBoard, add_noise=add_noise)

        # 在所有模拟结束后，处理可能剩余在批次中的节点
        if self.nodes_to_evaluate:
            self._evaluate_and_backpropagate()

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]

        if temp == 0:
            # 修复bug: 当所有访问计数都为0时，从有效移动中选择，而不是从所有移动中随机选择
            if np.max(counts) == 0:
                print("Warning: MCTS has no visit counts. Choosing first valid move.")
                valids = self.game.getValidMoves(canonicalBoard, 1)
                bestA = np.argmax(valids) # 确定性地选择第一个有效移动
            else:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)

            probs = np.zeros(len(counts))
            probs[bestA] = 1
            return probs

        counts_sum = float(sum(counts))
        if counts_sum == 0: # 极端情况下，如果所有走法计数都为0
            # 这种情况不应该频繁发生，但为了稳健性，返回一个基于有效走法的均匀分布
            valids = self.game.getValidMoves(canonicalBoard, 1)
            return valids / np.sum(valids)
            
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return np.array(probs)

    def search(self, canonicalBoard, add_noise=False):
        """
        执行一次MCTS模拟：从根节点遍历到叶节点，
        如果叶节点未被评估，则加入批处理队列。
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # --- 检查游戏是否在根节点就已结束 ---
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return # 游戏已结束，无需搜索

        path = []
        cur_board = canonicalBoard
        
        while True:
            s_cur = self.game.stringRepresentation(cur_board)
            
            # --- 选择阶段 ---
            if s_cur in self.Ps: # 如果不是叶节点
                valids = self.game.getValidMoves(cur_board, 1)
                cur_best = -float('inf')
                best_act = -1

                # PUCT算法
                for a in range(self.game.getActionSize()):
                    if valids[a]:
                        if (s_cur, a) in self.Qsa:
                            u = self.Qsa[(s_cur, a)] + self.args.cpuct * self.Ps[s_cur][a] * math.sqrt(self.Ns[s_cur]) / (1 + self.Nsa[(s_cur, a)])
                        else:
                            u = self.args.cpuct * self.Ps[s_cur][a] * math.sqrt(self.Ns[s_cur] + 1e-8)
                        
                        if u > cur_best:
                            cur_best = u
                            best_act = a
                
                a = best_act
                path.append((cur_board, a))
                
                cur_board, _ = self.game.getNextState(cur_board, 1, a)
                
                # 检查新状态是否结束
                s_next = self.game.stringRepresentation(cur_board)
                if s_next not in self.Es:
                     self.Es[s_next] = self.game.getGameEnded(cur_board, 1)
                
                if self.Es[s_next] != 0: # 如果下一步是终局
                    self._backpropagate(path, self.Es[s_next])
                    return
            else:
                # --- 扩展阶段 (延迟) ---
                # 到达叶节点，将其加入评估批次
                self.nodes_to_evaluate.append((cur_board, add_noise))
                self.paths_to_backprop.append(path)
                
                # 如果批次已满，则进行评估
                if len(self.nodes_to_evaluate) >= self.args.mcts_batch_size:
                    self._evaluate_and_backpropagate()
                return

    def _evaluate_and_backpropagate(self):
        """
        使用神经网络批量评估叶节点，然后反向传播价值。
        """
        if not self.nodes_to_evaluate:
            return

        # 立即复制并清空批处理列表，以避免在处理期间发生状态问题
        nodes_to_process = self.nodes_to_evaluate
        paths_to_process = self.paths_to_backprop
        self.nodes_to_evaluate = []
        self.paths_to_backprop = []

        # 批量预测
        boards_to_eval = [node[0] for node in nodes_to_process]
        policies, values = self.nnet.predict(boards_to_eval)

        for i, (board, should_add_noise) in enumerate(nodes_to_process):
            s = self.game.stringRepresentation(board)
            path = paths_to_process[i]
            pi = policies[i]
            v = values[i]

            # 对根节点策略添加狄利克雷噪声以鼓励探索
            if should_add_noise:
                valids = self.game.getValidMoves(board, 1)
                dir_noise = np.random.dirichlet([self.args.dirichlet_alpha] * np.sum(valids))
                
                full_dir_noise = np.zeros(len(pi))
                full_dir_noise[valids.astype(bool)] = dir_noise
                
                pi = pi * (1 - self.args.noise_epsilon) + full_dir_noise * self.args.noise_epsilon

            # 扩展叶节点
            valids = self.game.getValidMoves(board, 1)
            self.Ps[s] = pi * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # Fallback for rare cases where all valid moves have policy probability 0.
                # In this case, we assign a uniform probability to all valid moves.
                print("Warning: All valid moves were masked in MCTS expansion. Assigning uniform probability.")
                self.Ps[s] = valids / np.sum(valids)

            self.Ns[s] = 0
            
            # 反向传播价值
            self._backpropagate(path, v[0])

    def _backpropagate(self, path, v):
        """
        沿给定的路径反向传播价值。
        """
        # 注意：Dots and Boxes的价值更新逻辑与普通棋类不同
        # 1. 玩家视角在整个MCTS搜索中是固定的(player=1)
        # 2. 从叶节点返回的v是其父节点采取行动后，当前玩家的收益期望。
        #    因此，v的符号不需要在每一步都翻转。
        for board, a in reversed(path):
            s = self.game.stringRepresentation(board)
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1
            
            self.Ns[s] += 1
            
            # Dots & Boxes 的特殊规则：如果对手得分，我的收益是负的。
            # 这里传递的v已经处理了符号（来自 getGameEnded 或 nnet.predict)
            # 所以我们不需要在每一步都翻转v的符号
            # v = -v # 在这里不需要