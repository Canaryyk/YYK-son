# main.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from Coach import Coach
from config import args
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.pytorch.NNet import NNetWrapper as nn
import glob
import re

def main():
    print("Initializing game and neural network...")
    g = DotsAndBoxesGame(n=args.n)
    nnet = nn(g)
    
    loaded_model_iter = 0 # Default to 0, means from scratch or 'best.pth.tar' without a number
    if args.load_model:
        folder = args.load_folder_file[0]
        filename = args.load_folder_file[1]
        
        checkpoint_files = glob.glob(os.path.join(folder, "checkpoint_*.pth.tar"))
        latest_checkpoint_filename = None
        max_iter = -1
        
        if checkpoint_files:
            for f_path in checkpoint_files:
                f_name = os.path.basename(f_path)
                try:
                    iteration_str = re.search(r'checkpoint_(\d+)', f_name)
                    if iteration_str:
                        iteration = int(iteration_str.group(1))
                        if iteration > max_iter:
                            max_iter = iteration
                            latest_checkpoint_filename = f_name
                except (ValueError, IndexError):
                    continue
        
        if latest_checkpoint_filename:
            print(f"Found latest validated checkpoint: '{latest_checkpoint_filename}'. This will be loaded.")
            filename = latest_checkpoint_filename
            loaded_model_iter = max_iter
            args.load_folder_file = (folder, filename)
        else:
            print(f"No validated checkpoint found. Will attempt to load default: '{filename}'")

        model_path = os.path.join(folder, filename)
        if os.path.exists(model_path):
            print(f"Loading checkpoint from '{model_path}'...")
            nnet.load_checkpoint(folder, filename)
        else:
            print(f"Checkpoint file '{model_path}' not found. Starting training from scratch.")
            args.load_model = False # Don't try to load examples if model failed

    else:
        print("Starting training from scratch.")

    c = Coach(g, nnet, loaded_model_iter)

    if args.load_model:
        print("Attempting to load training examples...")
        c.load_train_examples()
    
    print("Starting the learning process...")
    c.learn()

if __name__ == "__main__":
    main()