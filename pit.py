# pit.py
import pygame
import numpy as np
import sys
from config import args
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.pytorch.NNet import NNetWrapper as NNet

# --- GUI 美化与增强 ---
# 更现代的颜色方案
BACKGROUND_COLOR = (248, 249, 250)
BOARD_BG_COLOR = (233, 236, 239)
LINE_COLOR = (134, 142, 150)
DOT_COLOR = (73, 80, 87)
HOVER_COLOR = (173, 216, 230, 200) # 淡蓝色，带透明度
PLAYER1_COLOR = (52, 152, 219) # 蓝色
PLAYER2_COLOR = (231, 76, 60)  # 红色
PLAYER1_FILL = (52, 152, 219, 150)
PLAYER2_FILL = (231, 76, 60, 150)
INFO_TEXT_COLOR = (52, 73, 94)
WIN_COLOR = (46, 204, 113)
LOSE_COLOR = (192, 57, 43)
DRAW_COLOR = (127, 140, 141)

# 尺寸和字体
CELL_SIZE = 70
MARGIN = 60
DOT_RADIUS = 7
LINE_THICKNESS = 6

class DotsAndBoxesGUI:
    """
    一个使用 Pygame 为点格棋游戏提供图形用户界面的类。
    """
    def __init__(self, game, nnet):
        self.g = game
        self.n = self.g.n
        self.nnet = nnet

        pygame.init()
        pygame.font.init()

        # 屏幕尺寸
        self.info_height = 120
        self.width = self.n * CELL_SIZE + 2 * MARGIN
        self.height = self.n * CELL_SIZE + 2 * MARGIN + self.info_height
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("点格棋 (Dots and Boxes)")

        # 字体
        try:
            # 尝试更现代的字体
            font_name = 'Microsoft YaHei UI'
            self.font = pygame.font.SysFont(font_name, 22)
            self.large_font = pygame.font.SysFont(font_name, 36)
            self.huge_font = pygame.font.SysFont(font_name, 60)
        except pygame.error:
            print(f"警告: '{font_name}' 字体未找到, 中文可能无法正常显示。将回退到默认字体。")
            self.font = pygame.font.Font(None, 28)
            self.large_font = pygame.font.Font(None, 45)
            self.huge_font = pygame.font.Font(None, 70)

        # 异步AI移动事件
        self.AI_MOVE_EVENT = pygame.USEREVENT + 1
        self.restart_game()

    def restart_game(self):
        """重置游戏到初始状态。"""
        self.board = self.g.getInitBoard()
        self.current_player = 1
        self.game_ended = 0
        self.hovered_action = None
        self.is_ai_thinking = False

    def get_action_from_pos(self, pos):
        """根据鼠标点击位置计算对应的动作ID。"""
        x, y = pos
        h_lines_count = (self.n + 1) * self.n
        
        # 为点击提供一个合理的容差范围
        click_tolerance = CELL_SIZE / 3.5

        # 检查水平线
        for r in range(self.n + 1):
            for c in range(self.n):
                line_x_start, line_y = MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE
                hitbox = pygame.Rect(line_x_start, line_y - click_tolerance / 2, CELL_SIZE, click_tolerance)
                if hitbox.collidepoint(x, y):
                    return r * self.n + c

        # 检查垂直线
        for r in range(self.n):
            for c in range(self.n + 1):
                line_x, line_y_start = MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE
                hitbox = pygame.Rect(line_x - click_tolerance / 2, line_y_start, click_tolerance, CELL_SIZE)
                if hitbox.collidepoint(x, y):
                    return h_lines_count + r * (self.n + 1) + c
        return None

    def draw_board(self):
        """绘制游戏棋盘和所有元素"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # 绘制棋盘背景
        board_rect = pygame.Rect(MARGIN, MARGIN, self.n * CELL_SIZE, self.n * CELL_SIZE)
        pygame.draw.rect(self.screen, BOARD_BG_COLOR, board_rect, border_radius=15)

        # 绘制已填充的方格
        _, _, boxes = self.board # 直接从board元组解包
        for r in range(self.n):
            for c in range(self.n):
                if boxes[r][c] != 0:
                    player = boxes[r][c]
                    color = PLAYER1_FILL if player == 1 else PLAYER2_FILL
                    
                    # 使用一个带圆角的Surface来绘制，效果更柔和
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(s, color, s.get_rect(), border_radius=10)
                    self.screen.blit(s, (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE))

        # 绘制所有线条
        h_lines, v_lines, _ = self.board
        for r in range(self.n + 1):
            for c in range(self.n):
                if h_lines[r][c]:
                    self.draw_line((MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE), 
                                   (MARGIN + (c + 1) * CELL_SIZE, MARGIN + r * CELL_SIZE))
        
        for r in range(self.n):
            for c in range(self.n + 1):
                if v_lines[r][c]:
                    self.draw_line((MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE),
                                   (MARGIN + c * CELL_SIZE, MARGIN + (r + 1) * CELL_SIZE))

        # 绘制悬停高亮效果
        self.draw_hover_effect()
        
        # 绘制棋盘上的点
        for r in range(self.n + 1):
            for c in range(self.n + 1):
                center = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
                pygame.draw.circle(self.screen, DOT_COLOR, center, DOT_RADIUS)

        self.draw_info()
        self.draw_game_over_overlay() # 绘制游戏结束浮层
        pygame.display.flip()
        
    def draw_line(self, start_pos, end_pos):
        pygame.draw.line(self.screen, LINE_COLOR, start_pos, end_pos, LINE_THICKNESS)
        pygame.draw.circle(self.screen, LINE_COLOR, start_pos, LINE_THICKNESS // 2)
        pygame.draw.circle(self.screen, LINE_COLOR, end_pos, LINE_THICKNESS // 2)

    def draw_hover_effect(self):
        if self.hovered_action is None or self.game_ended != 0:
            return

        valid_moves = self.g.getValidMoves(self.g.getCanonicalForm(self.board, self.current_player), 1)
        if not valid_moves[self.hovered_action]:
            return

        h_size = (self.n + 1) * self.n
        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        s.fill(HOVER_COLOR)
        
        if self.hovered_action < h_size:
            r, c = self.hovered_action // self.n, self.hovered_action % self.n
            pos = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE - LINE_THICKNESS // 2)
            rect = pygame.Rect(pos, (CELL_SIZE, LINE_THICKNESS))
        else:
            action = self.hovered_action - h_size
            r, c = action // (self.n + 1), action % (self.n + 1)
            pos = (MARGIN + c * CELL_SIZE - LINE_THICKNESS // 2, MARGIN + r * CELL_SIZE)
            rect = pygame.Rect(pos, (LINE_THICKNESS, CELL_SIZE))

        pygame.draw.rect(self.screen, HOVER_COLOR, rect, border_radius=LINE_THICKNESS // 2)
        
    def draw_info(self):
        """绘制分数和游戏状态信息"""
        # 分数显示
        score1 = self.g.getScore(self.board, 1)
        score2 = self.g.getScore(self.board, -1)
        score_text = f"你 {score1}  -  {score2} AI"  # noqa: F841
        
        p1_surf = self.large_font.render(f"你 {score1}", True, PLAYER1_COLOR)
        p2_surf = self.large_font.render(f"{score2} AI", True, PLAYER2_COLOR)
        
        self.screen.blit(p1_surf, p1_surf.get_rect(center=(self.width * 0.25, self.height - 75)))
        self.screen.blit(p2_surf, p2_surf.get_rect(center=(self.width * 0.75, self.height - 75)))

        # 回合提示 (仅在游戏进行时显示)
        if self.game_ended == 0:
            if self.is_ai_thinking:
                turn_text = "AI 正在思考..."
                color = INFO_TEXT_COLOR
            else:
                turn_text = "你的回合" if self.current_player == 1 else "等待AI..."
                color = PLAYER1_COLOR if self.current_player == 1 else PLAYER2_COLOR
            
            turn_surface = self.font.render(turn_text, True, color)
            self.screen.blit(turn_surface, turn_surface.get_rect(center=(self.width / 2, self.height - 35)))

    def draw_game_over_overlay(self):
        """当游戏结束时，绘制一个带有结果的浮层。"""
        if self.game_ended == 0:
            return

        # 创建一个半透明的浮层
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((248, 249, 250, 200))

        # 决定要显示的消息
        if self.game_ended == 1:
            end_text = "恭喜你，你赢了！"
            color = WIN_COLOR
        elif self.game_ended == -1:
            end_text = "AI 获胜！"
            color = LOSE_COLOR
        else: # 平局
            end_text = "平局！"
            color = DRAW_COLOR
        
        # 渲染主消息
        text_surface = self.huge_font.render(end_text, True, color)
        text_rect = text_surface.get_rect(center=(self.width / 2, self.height / 2 - 40))
        overlay.blit(text_surface, text_rect)

        # 渲染重新开始的提示
        restart_text = "按 R 键重新开始, 或按 ESC 键退出"
        restart_surface = self.font.render(restart_text, True, INFO_TEXT_COLOR)
        restart_rect = restart_surface.get_rect(center=(self.width / 2, self.height / 2 + 40))
        overlay.blit(restart_surface, restart_rect)

        self.screen.blit(overlay, (0, 0))

    def ai_move(self):
        """执行AI的移动"""
        if self.game_ended != 0 or self.current_player != -1: 
            print("AI move not allowed")
            return

        self.is_ai_thinking = True
        self.draw_board() # Re-draw to show "AI is thinking" message

        pygame.time.delay(100) 

        canonical_board = self.g.getCanonicalForm(self.board, self.current_player)
        pi, _ = self.nnet.predict(canonical_board)
        valid_moves = self.g.getValidMoves(canonical_board, 1)
        pi = pi * valid_moves
        
        action = np.argmax(pi)
        
        self.board, self.current_player = self.g.getNextState(self.board, -1, action)
        self.game_ended = self.g.getGameEnded(self.board, 1)
        self.is_ai_thinking = False
        
        # If AI scores and gets another turn, schedule the next move
        if self.game_ended == 0 and self.current_player == -1:
            pygame.time.set_timer(self.AI_MOVE_EVENT, 500, 1)
        
    def run(self):
        """游戏主循环 (已重构为单个事件循环)"""
        clock = pygame.time.Clock()
        running = True

        while running:
            # 单一事件处理循环
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if self.game_ended != 0 and event.key == pygame.K_r:
                        self.restart_game()
                
                # 仅当轮到玩家且游戏未结束时，处理鼠标点击
                if self.game_ended == 0:
                    if event.type == pygame.MOUSEMOTION and not self.is_ai_thinking:
                        self.hovered_action = self.get_action_from_pos(event.pos)

                    if self.current_player == 1 and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        action = self.get_action_from_pos(event.pos)
                        if action is not None:
                            valid_moves = self.g.getValidMoves(self.g.getCanonicalForm(self.board, 1), 1)
                            if valid_moves[action]:
                                self.board, self.current_player = self.g.getNextState(self.board, 1, action)
                                self.game_ended = self.g.getGameEnded(self.board, 1)
                                self.hovered_action = None # Clear hover after move

                                if self.game_ended == 0 and self.current_player == -1:
                                    pygame.time.set_timer(self.AI_MOVE_EVENT, 500, 1)
                    
                    if event.type == self.AI_MOVE_EVENT:
                        self.ai_move()

            # 绘制棋盘和所有UI元素
            self.draw_board()
            clock.tick(60) # Increased FPS for smoother hover animations

        pygame.quit()
        sys.exit()

def play_game_with_gui():
    g = DotsAndBoxesGame(n=args.n)
    n1 = NNet(g)
    
    try:
        n1.load_checkpoint(args.checkpoint, 'best.pth.tar')
        print("已加载 'best.pth.tar' 模型。")
    except FileNotFoundError:
        print(f"找不到 'best.pth.tar'。请确保模型文件在 '{args.checkpoint}' 目录下。")
        return

    gui = DotsAndBoxesGUI(game=g, nnet=n1)
    gui.run()

if __name__ == '__main__':
    play_game_with_gui()