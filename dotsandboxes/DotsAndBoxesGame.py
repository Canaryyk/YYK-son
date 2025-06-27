# dotsandboxes/DotsAndBoxesGame.py
import numpy as np

class DotsAndBoxesGame():
    """
    点格棋游戏逻辑实现。
    棋盘大小为 n x n 个盒子，即 (n+1) x (n+1) 个点。
    本项目中 n=7 (对应8x8点阵)。
    """
    def __init__(self, n=7):
        self.n = n

    def getInitBoard(self):
        """
        返回初始棋盘状态。
        棋盘状态是一个元组: (horizontal_edges, vertical_edges, boxes)。
        horizontal_edges: (n+1, n) 形状的数组。
        vertical_edges: (n, n+1) 形状的数组。
        boxes: (n, n) 形状的数组, 0-无归属, 1-玩家1, -1-玩家2。
        """
        h_edges = np.zeros((self.n + 1, self.n), dtype=np.int8)
        v_edges = np.zeros((self.n, self.n + 1), dtype=np.int8)
        boxes = np.zeros((self.n, self.n), dtype=np.int8)
        return (h_edges, v_edges, boxes)

    def getBoardSize(self):
        """ (水平边数, 垂直边数) """
        return ((self.n + 1) * self.n, self.n * (self.n + 1))

    def getActionSize(self):
        """ 总边数 """
        return 2 * self.n * (self.n + 1)

    def getNextState(self, board, player, action):
        """
        执行动作并返回下一个状态 (board, next_player)。
        核心逻辑：处理得分和额外回合。
        """
        # 创建棋盘数组的副本，以避免修改原始状态
        h_edges, v_edges, boxes = [np.copy(x) for x in board]

        newly_captured_boxes = 0
        h_size = (self.n + 1) * self.n

        if action < h_size:
            # 水平边
            row, col = action // self.n, action % self.n
            if h_edges[row, col]:
                # 此步无效，正常流程中不应发生
                return None, None
            h_edges[row, col] = 1

            # 检查新形成的盒子
            # 检查上方的盒子
            if row > 0 and v_edges[row - 1, col] and v_edges[row - 1, col + 1] and h_edges[row - 1, col]:
                if boxes[row - 1, col] == 0:
                    boxes[row - 1, col] = player
                    newly_captured_boxes += 1
            # 检查下方的盒子
            if row < self.n and v_edges[row, col] and v_edges[row, col + 1] and h_edges[row + 1, col]:
                if boxes[row, col] == 0:
                    boxes[row, col] = player
                    newly_captured_boxes += 1
        else:
            # 垂直边
            action -= h_size
            row, col = action // (self.n + 1), action % (self.n + 1)
            if v_edges[row, col]:
                # 此步无效，正常流程中不应发生
                return None, None
            v_edges[row, col] = 1

            # 检查新形成的盒子
            # 检查左侧的盒子
            if col > 0 and h_edges[row, col - 1] and h_edges[row + 1, col - 1] and v_edges[row, col - 1]:
                if boxes[row, col - 1] == 0:
                    boxes[row, col - 1] = player
                    newly_captured_boxes += 1
            # 检查右侧的盒子
            if col < self.n and h_edges[row, col] and h_edges[row + 1, col] and v_edges[row, col + 1]:
                if boxes[row, col] == 0:
                    boxes[row, col] = player
                    newly_captured_boxes += 1

        next_player = player if newly_captured_boxes > 0 else -player
        
        return ((h_edges, v_edges, boxes), next_player)

    def getValidMoves(self, board, player):
        """ 返回一个二值向量，表示所有合法动作。 """
        h_edges, v_edges, _ = board
        valid_moves = np.zeros(self.getActionSize(), dtype=np.uint8)
        
        h_flat = h_edges.flatten()
        v_flat = v_edges.flatten()
        
        valid_h = 1 - h_flat
        valid_v = 1 - v_flat
        
        valid_moves[:len(valid_h)] = valid_h
        valid_moves[len(valid_h):] = valid_v
        
        return valid_moves

    def getGameEnded(self, board, player):
        """
        判断游戏是否结束。
        如果结束，返回获胜方（1或-1）或平局（1e-4）。
        如果未结束，返回0。
        """
        h_edges, v_edges, boxes = board
        if np.all(h_edges) and np.all(v_edges):
            player1_score = np.sum(boxes == 1)
            player2_score = np.sum(boxes == -1)
            if player1_score > player2_score:
                return 1
            elif player2_score > player1_score:
                return -1
            else:
                return 1e-4  # 平局
        return 0

    def getCanonicalForm(self, board, player):
        """
        将棋盘转换为当前玩家的规范视角。
        player 1的视角是默认的, player -1的视角需要翻转box的归属。
        """
        h_edges, v_edges, boxes = board
        return (h_edges, v_edges, boxes * player)

    def getSymmetries(self, board, pi):
        """
        利用棋盘的对称性进行数据增强。
        点格棋盘有8种对称性（旋转和翻转）。
        """
        h_edges, v_edges, boxes = board
        pi_h_size = self.n * (self.n + 1)
        pi_h = pi[:pi_h_size].reshape(self.n + 1, self.n)
        pi_v = pi[pi_h_size:].reshape(self.n, self.n + 1)
        
        symmetries = []
        
        # 使用临时变量存储棋盘状态
        b_h, b_v, b_b = h_edges, v_edges, boxes
        p_h, p_v = pi_h, pi_v

        for _ in range(4):  # 4次旋转
            # 添加当前状态（恒等变换和3次旋转）
            symmetries.append(((b_h, b_v, b_b), np.concatenate((p_h.flatten(), p_v.flatten()))))

            # 添加水平翻转后的状态
            fh, fv, fb = np.fliplr(b_h), np.fliplr(b_v), np.fliplr(b_b)
            fph, fpv = np.fliplr(p_h), np.fliplr(p_v)
            symmetries.append(((fh, fv, fb), np.concatenate((fph.flatten(), fpv.flatten()))))

            # 为下一次迭代将状态顺时针旋转90度
            
            # 关键修正：将盒子的旋转方向从逆时针（k=1）修正为顺时针（k=-1），
            # 以匹配边（edges）和策略（policy）的旋转逻辑。
            b_b = np.rot90(b_b, k=-1)

            # 边和策略的旋转
            # 在覆盖前存储旧值
            b_h_old, b_v_old = b_h, b_v
            p_h_old, p_v_old = p_h, p_v
            
            # 新的水平边来自于旧的垂直边（顺时针旋转）
            b_h = np.rot90(b_v_old, k=-1)
            p_h = np.rot90(p_v_old, k=-1)

            # 新的垂直边来自于旧的水平边（顺时针旋转）
            b_v = np.rot90(b_h_old, k=-1)
            p_v = np.rot90(p_h_old, k=-1)
            
        # 添加断言来验证旋转后的形状
        for i, ((h, v, b), p) in enumerate(symmetries):
            assert h.shape == (self.n + 1, self.n), f"Invalid h_edges shape at symmetry {i}: {h.shape}"
            assert v.shape == (self.n, self.n + 1), f"Invalid v_edges shape at symmetry {i}: {v.shape}"
            assert b.shape == (self.n, self.n), f"Invalid boxes shape at symmetry {i}: {b.shape}"
            assert len(p) == self.getActionSize(), f"Invalid policy shape at symmetry {i}: {len(p)}"
            
        return symmetries

    def stringRepresentation(self, board):
        """ 将棋盘状态转换为唯一的字符串表示，用于MCTS的字典键。"""
        h_edges, v_edges, boxes = board
        return h_edges.tobytes() + v_edges.tobytes() + boxes.tobytes()

    def getScore(self, board, player):
        """ 获取指定玩家的分数。 """
        _, _, boxes = board
        return np.sum(boxes == player)

    def display(self, board):
        """ 在控制台打印棋盘。 """
        h_edges, v_edges, boxes = board
        n = self.n
        print("   ", end="")
        for c in range(n):
            print(f" {c}  ", end="")
        print("\n")

        for r in range(n + 1):
            # 打印点和水平边
            print(f"{r}  ", end="")
            for c in range(n + 1):
                print("o", end="")
                if c < n:
                    print("---" if h_edges[r, c] else "   ", end="")
            print("")

            if r < n:
                # 打印垂直边和盒子
                print("   ", end="")
                for c in range(n + 1):
                    print("|" if v_edges[r, c] else " ", end="  ")
                    if c < n:
                        box_val = boxes[r, c]
                        owner = " "
                        if box_val == 1:
                            owner = "1"
                        elif box_val == -1:
                            owner = "2"
                        print(owner, end=" ")
                print("")
        print("-" * (4 * n + 3))

