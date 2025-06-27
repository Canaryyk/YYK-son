# pit_models.py
import os
import sys
import multiprocessing
import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from config import args, dotdict
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.pytorch.NNet import NNetWrapper as NNet

"""
一个用于比较两个指定模型性能的脚本。
直接在此文件中配置模型路径和比赛参数，然后运行即可。
"""

def play_one_game(models_and_args):
    """
    在一个独立的进程中，让两个模型进行一局比赛。
    这个函数是并行化的基础。
    """
    model1_path, model2_path, game_args = models_and_args
    game = DotsAndBoxesGame(game_args.n) # 使用config中的n值

    # 使用def定义player，修复lambda赋值告警
    def create_player(model_path):
        net = NNet(game)
        net.load_checkpoint(folder=os.path.dirname(model_path), filename=os.path.basename(model_path))
        mcts = MCTS(game, net, game_args)
        return lambda x: np.argmax(mcts.getActionProb(x, temp=0, add_noise=False))

    player1 = create_player(model1_path)
    player2 = create_player(model2_path)

    players = {1: player1, -1: player2}
    current_player = 1
    board = game.getInitBoard()
    
    while game.getGameEnded(board, 1) == 0:
        canonical_board = game.getCanonicalForm(board, current_player)
        action = players[current_player](canonical_board)
        
        valids = game.getValidMoves(canonical_board, 1)
        if valids[action] == 0:
            print(f"FATAL: Player {current_player} played an invalid move.")
            assert valids[action] > 0
        
        # 修复pyright告警: 显式处理getNextState可能返回None的情况
        board, next_player = game.getNextState(board, current_player, action)
        if next_player is None:
            # 这个分支理论上因为上面的assert而无法到达，但它可以让代码更健壮并消除告警
            print(f"FATAL: Game state became invalid for player {current_player}. Ending game.")
            break
        current_player = next_player
    
    return game.getGameEnded(board, 1)

def main():
    # ----- 在这里配置你要对弈的模型 -----
    MODEL_1_PATH = './temp/checkpoint_27.pth.tar'
    MODEL_2_PATH = './temp/checkpoint_24.pth.tar'
    NUM_GAMES = 50
    # ------------------------------------

    # 修复dotdict无法赋值的bug
    # 将游戏参数从主config中复制一份为普通字典，以安全地修改
    game_args_dict = dict(args) 
    # 在竞技场中，我们通常使用更多的MCTS模拟来获得更准确的棋力评估
    game_args_dict['num_mcts_sims'] = 50
    game_args_dict['arena_num_mcts_sims'] = 50 # 也更新这个，保持一致
    game_args_dict['mcts_batch_size'] = 32 

    # 将修改后的字典转回dotdict，以便MCTS等模块使用
    game_args = dotdict(game_args_dict)

    model1_name = os.path.basename(MODEL_1_PATH)
    model2_name = os.path.basename(MODEL_2_PATH)

    print(f"Pitting {model1_name} vs {model2_name} for {NUM_GAMES} games.")

    # 设置多进程启动方式，保证跨平台兼容性
    if sys.platform in ['win32', 'cygwin', 'darwin']:
        multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(args.num_workers) as pool:
        model1_wins, model2_wins, draws = 0, 0, 0
        
        # --- 第一轮: model1执先手 ---
        print(f"Round 1: {model1_name} as Player 1")
        round1_args = [(MODEL_1_PATH, MODEL_2_PATH, game_args)] * (NUM_GAMES // 2)
        results = list(tqdm(pool.imap(play_one_game, round1_args), total=len(round1_args), desc=f"{model1_name} as P1"))
        model1_wins += results.count(1)
        model2_wins += results.count(-1)
        draws += results.count(1e-4)

        # --- 第二轮: model2执先手 ---
        print(f"Round 2: {model2_name} as Player 1")
        round2_args = [(MODEL_2_PATH, MODEL_1_PATH, game_args)] * (NUM_GAMES - len(round1_args))
        results_swapped = list(tqdm(pool.imap(play_one_game, round2_args), total=len(round2_args), desc=f"{model2_name} as P1"))
        model2_wins += results_swapped.count(1)
        model1_wins += results_swapped.count(-1)
        draws += results_swapped.count(1e-4)

    print("\n----- FINAL RESULTS -----")
    print(f"Total Games: {NUM_GAMES}")
    print(f"Wins for {model1_name}: {model1_wins}")
    print(f"Wins for {model2_name}: {model2_wins}")
    print(f"Draws: {draws}")
    
    if NUM_GAMES > 0:
        win_rate_model1 = (model1_wins + 0.5 * draws) / NUM_GAMES
        print(f"\nWin Rate for {model1_name} (draws as 0.5 wins): {win_rate_model1:.2%}")

if __name__ == "__main__":
    main() 