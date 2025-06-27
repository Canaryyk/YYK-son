# dotsandboxes/pytorch/NNet.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import args
from torch.utils.data import Dataset, DataLoader

class ResBlock(nn.Module):
    """ 单个残差块的实现。 """
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DotsAndBoxesNNet(nn.Module):
    """
    神经网络模型，包含一个ResNet主干和策略/价值双头输出。
    """
    def __init__(self, game, args):
        super(DotsAndBoxesNNet, self).__init__()
        self.n = game.n
        self.action_size = game.getActionSize()
        self.args = args

        # 输入尺寸: (batch_size, 4, n+1, n+1)
        # 通道0: 水平边, 通道1: 垂直边, 通道2: 我方盒子, 通道3: 对方盒子
        input_board_size = self.n + 1 # 使用 n+1 x n+1 的维度以自然地表示边
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(4, args.num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU()
        )
        
        # 参照报告，残差塔由多个残差块构成
        self.res_tower = nn.Sequential(
            *[ResBlock(args.num_channels) for _ in range(args.num_res_blocks)]
        )

        # 策略头
        self.policy_head_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, stride=1)
        self.policy_head_bn = nn.BatchNorm2d(2)
        self.policy_head_fc = nn.Linear(2 * input_board_size * input_board_size, self.action_size)

        # 价值头
        self.value_head_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, stride=1)
        self.value_head_bn = nn.BatchNorm2d(1)
        self.value_head_fc1 = nn.Linear(1 * input_board_size * input_board_size, 256)
        self.value_head_fc2 = nn.Linear(256, 1)

        self.fc_drop = nn.Dropout(p=self.args.dropout)

    def forward(self, s):
        s = s.view(-1, 4, self.n + 1, self.n + 1)  # 确保输入形状正确
        s = self.conv_in(s)
        s = self.res_tower(s)
        
        # 策略头
        pi = self.policy_head_conv(s)
        pi = F.relu(self.policy_head_bn(pi))
        pi = pi.view(-1, 2 * (self.n + 1) * (self.n + 1))
        pi = self.fc_drop(pi)
        pi = self.policy_head_fc(pi)
        
        # 价值头
        v = self.value_head_conv(s)
        v = F.relu(self.value_head_bn(v))
        v = v.view(-1, 1 * (self.n + 1) * (self.n + 1))
        v = F.relu(self.value_head_fc1(v))
        v = self.fc_drop(v)
        v = torch.tanh(self.value_head_fc2(v))
        
        return F.log_softmax(pi, dim=1), v

class NNetWrapper():
    """
    包装器类，提供统一的训练、预测、存取模型接口。
    """
    def __init__(self, game):
        self.nnet = DotsAndBoxesNNet(game, args)
        self.n = game.n
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        if args.cuda:
            self.nnet.cuda()

    def _state_to_tensor(self, board):
        """
        辅助函数：将游戏逻辑中的board元组转换为4D张量。
        这个版本会处理旋转导致的形状变化。
        """
        h_edges, v_edges, boxes = board
        tensor = np.zeros((4, self.n + 1, self.n + 1), dtype=np.float32)
        
        # After the fix in getSymmetries, the shapes of h_edges and v_edges are always consistent.
        # h_edges is (n+1, n) and v_edges is (n, n+1). The if/else is no longer needed.
        tensor[0, :, :self.n] = h_edges
        tensor[1, :self.n, :] = v_edges

        # boxes' shape is (n,n), padding to (n+1, n+1) is implicit.
        tensor[2, :self.n, :self.n] = (boxes == 1)
        tensor[3, :self.n, :self.n] = (boxes == -1)
        
        return torch.FloatTensor(tensor)
        
    def train(self, examples):
        """
        训练神经网络。
        examples: 从经验回放池中采样的 (board, pi, v) 列表。
        """
        train_dataset = TrainDataset(examples, self)
        
        # 使用DataLoader实现多进程数据加载和预处理
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers, # 利用多核CPU
            pin_memory=True # 加速数据到GPU的传输
        )
        
        # 引入AMP GradScaler
        scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = []
            v_losses = []

            t = tqdm(train_loader, desc='Training Net', file=sys.stdout)
            for boards, target_pis, target_vs in t:
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(non_blocking=True), target_pis.contiguous().cuda(non_blocking=True), target_vs.contiguous().cuda(non_blocking=True)

                # 使用autocast上下文管理器
                with torch.cuda.amp.autocast(enabled=args.cuda):
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v
                
                # --- 关键检查: 检查损失是否有效 ---
                if not torch.isfinite(total_loss):
                    raise ValueError(f"Loss is {total_loss}, stopping training. Check learning rate or data integrity.")

                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                t.set_postfix(Loss_pi=np.mean(pi_losses), Loss_v=np.mean(v_losses))

                self.optimizer.zero_grad()
                # 使用scaler进行反向传播和参数更新
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            # 在epoch结束时打印最终的平均损失
            final_pi_loss = np.mean(pi_losses)
            final_v_loss = np.mean(v_losses)
            print(f"End of Epoch {epoch+1}: Avg Policy Loss = {final_pi_loss:.4f}, Avg Value Loss = {final_v_loss:.4f}")

    def predict(self, boards):
        """
        对给定的board列表进行预测。
        返回策略向量p和价值v的列表。
        """
        if not isinstance(boards, list):
            boards = [boards]

        board_tensors = torch.stack([self._state_to_tensor(b) for b in boards])

        if args.cuda:
            board_tensors = board_tensors.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_tensors)

        pis = torch.exp(pi).data.cpu().numpy()
        vs = v.data.cpu().numpy()

        return pis, vs

    def loss_pi(self, targets, outputs):
        """ 策略损失：交叉熵。 """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """ 价值损失：均方误差。 """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = 'cuda:0' if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        
        # 加载优化器状态，如果存在的话
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: Optimizer state not found in checkpoint. Initializing a new optimizer.")

class TrainDataset(Dataset):
    """
    一个简单的PyTorch数据集，用于包装训练样本。
    """
    def __init__(self, examples, nnet_wrapper):
        self.examples = examples
        self.nnet_wrapper = nnet_wrapper

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        board, pi, v = self.examples[idx]
        board_tensor = self.nnet_wrapper._state_to_tensor(board)
        pi_tensor = torch.FloatTensor(np.array(pi))
        v_tensor = torch.tensor(v, dtype=torch.float32) # v是标量
        return board_tensor, pi_tensor, v_tensor