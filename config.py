# config.py
# 存放所有超参数

import os


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
            
    def __setattr__(self, name, value):  # <--- 添加这个方法
        self[name] = value


args = dotdict({
    # ======================================================================================
    # === 高质量与速度均衡配置 (Balanced Quality & Speed) ===
    # ======================================================================================
    'num_iterations': 2000,          # 总的"自我对弈-训练-评估"循环次数 (可以保持不变或根据需要增加)
    'num_eps': 40,                  # 每次迭代中进行的自我对弈局数 (适当增加以获得更多样化的数据)
    'num_workers': 8,               # 并行执行自我对弈的工作进程数
    'temp_threshold': 15,           # 在自我对弈的前15步，使用温度参数tau=1以鼓励探索 (保持不变)
    'update_threshold': 0.55,       # 新模型取代旧模型所需的胜率阈值 
    'max_len_of_queue': 700000,     # 经验回放池的最大长度 (增加以容纳更多历史数据，防止遗忘)
    'num_mcts_sims': 500,           # 每一步棋执行的MCTS模拟次数 (保持不变，这是高质量数据的保证)
    'mcts_batch_size': 64,            # MCTS批量预测时的批次大小
    'arena_compare': 40,            # 在竞技场中比较新旧模型的比赛局数 (增加对局数，减少随机性)
    'arena_num_mcts_sims': 500,     # Arena对战时的MCTS模拟次数 (***核心修正***: 必须与训练时的模拟次数相同)
    'cpuct': 1.5,                   # PUCT公式中的探索常数 (稍微降低，在网络变强后可以更相信其策略)
    'dirichlet_alpha': 0.3,         # 用于为根节点策略添加探索噪声的Dirichlet分布的alpha参数
    'noise_epsilon': 0.25,          # 噪声在根节点策略中所占的比例

    'checkpoint': './temp/',        # 模型检查点保存路径
    'load_model': True,           
    'load_folder_file': ('./temp/','best.pth.tar'),

    'n': 7,  # 棋盘大小

    # 神经网络参数
    'lr': 0.001,                   # Adam优化器的学习率 (***核心修正***: 大幅降低以匹配更稳定的训练)
    'dropout': 0.3,                  # Dropout比率
    'epochs': 1,                     # 每次迭代中，对采样出的训练数据进行的训练轮数 (***核心修正***: 改为1，防止过拟合)
    'batch_size': 128,               # 训练时的批次大小 (在GPU显存允许的情况下可以增大，以获得更稳定的梯度)
    'cuda': True,                    # 是否使用GPU
    'num_channels': 128,             # ResNet残差块中的卷积核数量 (***核心修正***: 增加网络宽度)
    'num_res_blocks': 7,            # ResNet残差块的数量 (***核心修正***: 增加网络深度)
})
