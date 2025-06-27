# Coach.py
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch # 引入torch用于权重比较
import time
from MCTS import MCTS
# Arena is no longer used directly for parallel execution
# from Arena import Arena 
from config import args

def execute_episode_wrapper(args_dict):
    """
    一个包装函数，用于在多进程中执行单局自我对弈。
    A wrapper function to run a single episode of self-play in a multiprocessing setup.
    """
    game_class = args_dict['game_class']
    # nnet is not passed directly. Instead, the worker creates its own instance
    # and loads the weights from a checkpoint file.
    # nnet = args_dict['nnet'] 
    args_config = args_dict['args']
    
    game = game_class()
    # The NNetWrapper class needs to be imported or available in this scope
    from dotsandboxes.pytorch.NNet import NNetWrapper as NNet
    nnet = NNet(game)
    nnet.load_checkpoint(folder=args_config.checkpoint, filename='temp.pth.tar')

    mcts = MCTS(game, nnet, args_config)
    
    train_examples = []
    board = game.getInitBoard()
    current_player = 1
    episode_step = 0

    while True:
        episode_step += 1
        canonical_board = game.getCanonicalForm(board, current_player)
        temp = int(episode_step < args_config.temp_threshold)

        pi = mcts.getActionProb(canonical_board, temp=temp, add_noise=True)
        sym = game.getSymmetries(canonical_board, pi)
        for b, p in sym:
            train_examples.append([b, current_player, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.getNextState(board, current_player, action)
        
        r = game.getGameEnded(board, current_player)

        if r != 0:
            # 为所有步骤分配最终结果v
            return [(x[0], x[2], r * ((-1) ** (x[1] != current_player))) for x in train_examples]

def arena_worker(args_pack):
    """
    一个用于在单个进程中进行一局比赛的worker函数。
    A worker function to play a single game in a process.
    """
    game_class, p1_path, p2_path, args_config = args_pack
    game = game_class()
    from dotsandboxes.pytorch.NNet import NNetWrapper as NNet

    p1_net = NNet(game)
    p1_net.load_checkpoint(folder=os.path.dirname(p1_path), filename=os.path.basename(p1_path))
    p2_net = NNet(game)
    p2_net.load_checkpoint(folder=os.path.dirname(p2_path), filename=os.path.basename(p2_path))

    p1_mcts = MCTS(game, p1_net, args_config)
    p2_mcts = MCTS(game, p2_net, args_config)

    # Use def for clarity as suggested by linters
    def player1(x):
        return np.argmax(p1_mcts.getActionProb(x, temp=0, add_noise=False))
    def player2(x):
        return np.argmax(p2_mcts.getActionProb(x, temp=0, add_noise=False))

    players = {1: player1, -1: player2}
    current_player = 1
    board = game.getInitBoard()
    
    while game.getGameEnded(board, 1) == 0:
        canonical_board = game.getCanonicalForm(board, current_player)
        action = players[current_player](canonical_board)
        
        valids = game.getValidMoves(canonical_board, 1)
        # --- 关键检查: 检查竞技场中的非法移动 ---
        assert valids[action] > 0, "FATAL: Arena player made an invalid move."
        board, current_player = game.getNextState(board, current_player, action)
    
    return game.getGameEnded(board, 1)

class Coach():
    """
    这个类执行完整的训练循环。
    它管理自我对弈、学习、评估和经验回放池。
    This class executes the complete training loop. It orchestrates self-play,
    learning, and evaluation.
    """
    def __init__(self, game, nnet, last_iter=0):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.train_examples_history = deque(maxlen=args.max_len_of_queue)
        self.start_iter = last_iter + 1

    def learn(self):
        """
        Performs num_iterations iterations with num_eps episodes of self-play in each
        iteration. After every iteration, it retrains the network with
        examples in train_examples_history (which has a maximum length of
        num_iters_for_train_examples_history). It then pits the new network against the
        old network and accepts it only if it wins >= update_threshold fraction of
        games.
        """

        for i in range(self.start_iter, args.num_iterations + 1):
            print(f'------ ITERATION {i} ------')
            # self.skip_first_self_play an alpha zero variant logic that skips self-play if examples are loaded
            # some alpha zero variants load examples from a file and skip self-play for the first iteration.
                # 在此迭代中生成的训练数据
            iteration_train_examples = deque()
            
            #
            # Parallelized self-play
            #
            print("Starting Self-Play...")
            start_time = time.time()
            # First, save the current network to be loaded by workers
            self.nnet.save_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar') # pnet is now the same as nnet

            args_dict = {
                'game_class': self.game.__class__,
                # 'nnet': self.nnet, # Do not pass the network object
                'args': args
            }
            
            print(f"Executing {args.num_eps} episodes with {args.num_workers} workers...")

            # Set start method for multiprocessing on Windows and macOS
            if sys.platform in ['win32', 'cygwin', 'darwin']:
                multiprocessing.set_start_method('spawn', force=True)

            with multiprocessing.Pool(args.num_workers) as pool:
                # 使用tqdm直接包装pool.imap，实现流式处理
                pbar = tqdm(pool.imap(execute_episode_wrapper, [args_dict] * args.num_eps), total=args.num_eps, desc="Self Play", file=sys.stdout)
                for result in pbar:
                    iteration_train_examples.extend(result)

            self_play_duration = time.time() - start_time
            print(f"Self-Play finished in {self_play_duration:.2f}s.")
            
            # 将新生成的例子添加到主经验回放池中
            self.train_examples_history.extend(iteration_train_examples)
            
            print(f"Data Generation: {len(iteration_train_examples)} new examples added.")
            print(f"Experience Buffer: Total size is {len(self.train_examples_history)}/{args.max_len_of_queue}.")

            # --- 关键修复: 保证模型和样本文件编号一致 ---
            self.save_train_examples(i)
        
            # --- 关键检查: 训练前后权重对比 ---
            # 1. 深度复制训练前的模型状态
            pre_train_state_dict = {k: v.clone() for k, v in self.nnet.nnet.state_dict().items()}

            # train the network
            print("Starting Training...")
            start_time = time.time()
            # At this point, self.nnet is the "champion" from the previous iteration.
            # We train it further, turning it into the "challenger".
            self.nnet.train(self.train_examples_history)
            train_duration = time.time() - start_time
            print(f"Training finished in {train_duration:.2f}s.")
            
            # 2. 比较训练后的状态
            post_train_state_dict = self.nnet.nnet.state_dict()
            weights_changed = any(not torch.equal(pre_train_state_dict[k], post_train_state_dict[k]) for k in pre_train_state_dict)
            if not weights_changed:
                print("WARNING: Model weights did not change after training. The model might not be learning.")

            self.nnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
            # self.pnet is already loaded with the "champion" state from before training.

            print('PITTING AGAINST PREVIOUS VERSION (Parallelized)')
            start_time = time.time()
            
            pnet_path = os.path.join(args.checkpoint, 'temp.pth.tar') # Champion
            nnet_path = os.path.join(args.checkpoint, 'best.pth.tar') # Challenger

            if sys.platform in ['win32', 'cygwin', 'darwin']:
                multiprocessing.set_start_method('spawn', force=True)

            with multiprocessing.Pool(args.num_workers) as pool:
                nwins, pwins, draws = 0, 0, 0
                
                # Round 1: nnet is player 1
                print("Pitting Round 1: New model as Player 1...")
                args_pack_list = [(self.game.__class__, nnet_path, pnet_path, args)] * args.arena_compare
                results_r1 = list(tqdm(pool.imap(arena_worker, args_pack_list), total=args.arena_compare, desc="Arena Games (nnet as p1)", file=sys.stdout))
                r1_nwins = results_r1.count(1)
                r1_pwins = results_r1.count(-1)
                r1_draws = results_r1.count(1e-4)
                print(f'Round 1 Results: New/Prev Wins : {r1_nwins} / {r1_pwins} ; Draws : {r1_draws}')
                
                nwins += r1_nwins
                pwins += r1_pwins
                draws += r1_draws

                # Round 2: pnet is player 1
                print("\nPitting Round 2: New model as Player 2...")
                args_pack_list_swapped = [(self.game.__class__, pnet_path, nnet_path, args)] * args.arena_compare
                results_r2 = list(tqdm(pool.imap(arena_worker, args_pack_list_swapped), total=args.arena_compare, desc="Arena Games (pnet as p1)", file=sys.stdout))
                # Here, a win for pnet (1) is a loss for nnet. A loss for pnet (-1) is a win for nnet.
                r2_pwins = results_r2.count(1)
                r2_nwins = results_r2.count(-1)
                r2_draws = results_r2.count(1e-4)
                print(f'Round 2 Results: New/Prev Wins : {r2_nwins} / {r2_pwins} ; Draws : {r2_draws}\n')
                
                pwins += r2_pwins
                nwins += r2_nwins
                draws += r2_draws
            
            pitting_duration = time.time() - start_time
            print(f"Pitting finished in {pitting_duration:.2f}s.")

            # --- 关键检查: 检查平局率是否过高 ---
            total_games = nwins + pwins + draws
            if total_games > 0:
                draw_rate = draws / total_games
                win_rate_new_model = nwins / (pwins + nwins) if (pwins + nwins) > 0 else 0
                print(f'TOTAL RESULTS: NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws} (Win Rate: {win_rate_new_model:.2%})')
                if draw_rate > 0.5 and total_games > 10: # 平局率阈值和最小对局数
                    print(f"WARNING: High draw rate detected ({draw_rate:.2%}). Models may have similar strength or be stuck in repetitive patterns.")

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) >= args.update_threshold:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=args.checkpoint, filename=self.get_checkpoint_file(i))
            else:
                print('REJECTING NEW MODEL')
                # Load back the old model weights from temp file
                self.nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
                self.nnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')

    def get_checkpoint_file(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def save_train_examples(self, iteration):
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        
    def load_train_examples(self):
        """
        一个统一的、健壮的加载训练样本的实现。
        它直接使用args中记录的、实际加载的模型文件路径来寻找对应的样本文件。
        """
        model_path = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
        examples_file = model_path + ".examples"

        if not os.path.isfile(examples_file):
            print(f"Warning: Examples file '{examples_file}' not found. Starting with a new experience buffer.")
            # 确保在找不到文件时，我们使用的是一个正确初始化的空队列
            self.train_examples_history = deque(maxlen=args.max_len_of_queue)
            return
            
        print(f"Loading examples from {examples_file}...")
        with open(examples_file, "rb") as f:
            loaded_examples = Unpickler(f).load()

        # 验证数据格式
        if loaded_examples and len(loaded_examples) > 0:
            sample = loaded_examples[0]
            if not isinstance(sample, (list, tuple)) or len(sample) != 3:  # (board, pi, v)
                print(f"Error: Invalid example format in {examples_file}. Expected (board, pi, v). Got: {sample}")
                print("Starting with a new experience buffer to ensure data consistency.")
                self.train_examples_history = deque(maxlen=args.max_len_of_queue)
                return
        
        self.train_examples_history = loaded_examples
        print(f"Resuming from iteration {self.start_iter}")