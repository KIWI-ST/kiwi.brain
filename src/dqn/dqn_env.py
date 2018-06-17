import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40   # pixels
FORM_H = 4  # grid height
FORM_W = 4  # grid width

WINDOW_W = 800
WINDOW_H = 600

class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        #定义GUI标题
        self.title('分类器')
        #定义分类动作
        self.action_space = ['yellow', 'red', 'back',
                             'white', 'gray', 'green', 'orange']
        #记录分类动作总数
        self.n_actions = len(self.action_space)
        #定义神经网络输入单元参数个数
        self.n_features = 10
        #设定窗口大小
        self.geometry('{0}x{1}'.format(WINDOW_W, WINDOW_H))
        #初始化
        self._build_maze()

    #初始化GUI窗口，用于可视化训练界面
    def _build_maze(self):
        self.canvas = 1
        #创建绘制画布
        # 
        # self.canvas = tk.Canvas(
        #     self, bg='white', height=FORM_H * UNIT, width=FORM_W * UNIT)
        # #绘制矩形格网
        # for c in range(0, FORM_W * UNIT, UNIT):
        #     x0, y0, x1, y1 = c, 0, c, FORM_H * UNIT
        #     self.canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, FORM_H * UNIT, UNIT):
        #     x0, y0, x1, y1 = 0, r, FORM_H * UNIT, r
        #     self.canvas.create_line(x0, y0, x1, y1)
        # # create origin
        # origin = np.array([20, 20])
        # #洞区
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black')
        # # hell
        # # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # # self.hell2 = self.canvas.create_rectangle(
        # #     hell2_center[0] - 15, hell2_center[1] - 15,
        # #     hell2_center[0] + 15, hell2_center[1] + 15,
        # #     fill='black')

        # # create oval
        # oval_center = origin + UNIT * 2
        # self.oval = self.canvas.create_oval(
        #     oval_center[0] - 15, oval_center[1] - 15,
        #     oval_center[0] + 15, oval_center[1] + 15,
        #     fill='yellow')

        # # create red rect
        # self.rect = self.canvas.create_rectangle(
        #     origin[0] - 15, origin[1] - 15,
        #     origin[0] + 15, origin[1] + 15,
        #     fill='red')
        # pack all
        # self.canvas.pack()

    def reset(self):
        self.render()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(FORM_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (FORM_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (FORM_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(
            self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(
            next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(FORM_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.render()


if __name__ == '__main__':
    #初始化环境
    Environment = Env()
    #执行UI消息监听
    Environment.mainloop()
    #更新UI
    Environment.render()
