import math

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np


class Vector2d():
    
    def __init__(self, x, y):
        self.deltaX = x        # 表示向量时，是x方向上的变化量。表示点时是起点是原点的向量
        self.deltaY = y
        self.length = -1       # 向量长度
        self.Unit_Vec = [0, 0] # 此向量的单位向量
        self.property_setting()

    def property_setting(self):
        '''
            设置向量的长度，单位向量
        '''
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
        if self.length > 0:
            self.Unit_Vec = [self.deltaX / self.length, self.deltaY / self.length]
        else:
            self.Unit_Vec = None

    def __add__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.property_setting()
        return vec

    def __sub__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.property_setting()
        return vec

    def __mul__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.property_setting()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __repr__(self):
        return 'Vector X:{}, Y:{}, length:{}, Unit_Vec:{}'.format(self.deltaX, self.deltaY, self.length,
                                                                             self.Unit_Vec)
        
class APF():
    def __init__(self, start: (), goal: (), obstacle_List: [], k_att: float, k_rep: float, rep_range: float,
                 step_size: float, max_iters: int, goal_threshold: float):
            
            self.start           = Vector2d(start[0], start[1]) # 起点vector
            self.goal            = Vector2d(goal[0], goal[1])   # 终点vector
            self.V_obstacle_list = [Vector2d(OB[0], OB[1]) for OB in obstacle_List] # 障碍物列表，每个元素为Vector2d对象
            self.k_att           = k_att          # 引力系数
            self.k_rep           = k_rep          # 斥力系数
            self.rep_range       = rep_range      # 斥力作用范围
            self.step_size       = step_size      # 步长
            self.max_iters       = max_iters      # 最大迭代次数
            self.goal_threashold = goal_threshold # 离目标点小于此值即认为到达目标点
            
            self.cur_iters            = 0                            # 当前迭代数
            self.current_pos          = Vector2d(start[0], start[1]) # 当前位置
            self.is_path_plan_success = False                        # 是否陷入不可达或局部最小值
            self.path                 = list()                       # 规划路径
            
            
    def attractive_F(self):
        '''
            U_att(cur) = 1/2 * α * |cur - goal|^2 
            F_att(cur) = - ▽U_att(cur) = - α * (cur - goal)
        '''
        att = (self.goal - self.current_pos) * self.k_att  
        return att
    
    def repulsion_F(self):
        """
            U_rep(cur) = 1/2 * β * (1 / (|cur - goal|) - 1 / rr)^2
            F_rep(cur) = β * (1 / (|cur - goal|) - 1 / rr) * 1/(|cur - goal|^2) * ▽(|cur - goal|)
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for ob in self.V_obstacle_list:
            rep_F = self.current_pos - ob # 矢量斥力
            if (rep_F.length <= self.rep_range):  # 在斥力影响范围
                rep += Vector2d(rep_F.Unit_Vec[0], rep_F.Unit_Vec[1]) * self.k_rep * (1.0 / rep_F.length - 1.0 / self.rep_range) / (rep_F.length ** 2) 
        return rep
    
    def path_plan(self):
        while (self.cur_iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threashold):
            f_vec = self.attractive_F() + self.repulsion_F()
            self.current_pos += Vector2d(f_vec.Unit_Vec[0], f_vec.Unit_Vec[1]) * self.step_size
            self.cur_iters   += 1
            self.path.append([self.current_pos.deltaX, self.current_pos.deltaY])

        if (self.current_pos - self.goal).length <= self.goal_threashold:
            self.is_path_plan_success = True
            
        # Local minima : modify the obstacle center and repulsive region
        if not self.is_path_plan_success:
            # The degree of change
            degree = 0.000000000001 
            
            modified_obstacle_List = []
            
            for ob in self.V_obstacle_list:
                if self.current_pos.deltaX / ob.Unit_Vec[0] - self.current_pos.deltaY / ob.Unit_Vec[1] < 1e-6:
                    ob.deltaX += degree
                modified_obstacle_List.append([ob.deltaX,ob.deltaY])
            
            modified_rep_range = self.rep_range + degree
            
            APF_modified = APF(start  = (self.start.deltaX,self.start.deltaY), 
               goal           = (self.goal.deltaX,self.goal.deltaY),
               obstacle_List  = modified_obstacle_List,
               k_att          = self.k_att, 
               k_rep          = self.k_rep,
               rep_range      = modified_rep_range, 
               step_size      = self.step_size,
               max_iters      = self.max_iters,
               goal_threshold = self.goal_threashold)
            return APF_modified
        print("Path planning is successfull: ", self.is_path_plan_success)

        
def filling(obstacles):
    xs = []
    ys = []
    for ob in obstacles:
        xs.append(ob[0])
    for ob in obstacles:
        ys.append(ob[1])    
    xs = np.array(xs)
    ys = np.array(ys)
    xs = np.unique(xs)
    ys = np.unique(ys)
    fill_obs = []
    for i in xs:
        for j in ys:             
            fill_obs.append([i,j])
    return fill_obs

if __name__ == '__main__':
    
    obstacles1 = [[1.5,3],[2, 3],[2.5, 3],[3, 3], [3, 2], [3, 2.5]]
    obstacles2 = [[1,1]]
    obstacles = obstacles1 + obstacles2
    rep_range      = 0.5
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
    fig = plt.figure(figsize=(14, 7))
    subplot = fig.add_subplot(121)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')

    
    obstacles1 = [[1.5,3],[2, 3],[2.5, 3],[3, 3], [3, 2], [3, 2.5]]
    obstacles1 = filling(obstacles1)
    obstacles2 = [[1,1]]
    obstacles = obstacles1 + obstacles2
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
#   fig = plt.figure(figsize=(7, 7))
    subplot = fig.add_subplot(122)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')
    plt.show()

if __name__ == '__main__':
    
    obstacles1 = [[2,2],[2.5,2],[2, 3],[2.5, 3],[3, 3], [3, 2], [3, 2.5]]
    obstacles2 = [[1,1]]
    obstacles = obstacles1 + obstacles2
    rep_range      = 0.5
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
    fig = plt.figure(figsize=(14, 7))
    subplot = fig.add_subplot(121)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')

    
    obstacles1 = [[2,2],[2.5,2],[2, 3],[2.5, 3],[3, 3], [3, 2], [3, 2.5]]
    obstacles1 = filling(obstacles1)
    obstacles2 = [[1,1]]
    obstacles = obstacles1 + obstacles2
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
#   fig = plt.figure(figsize=(7, 7))
    subplot = fig.add_subplot(122)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')
    plt.show()

if __name__ == '__main__':
    
    obstacles = [[2,2],[2.5,2],[1.75,2.25],[2.75,2.25],[2.25,2.5]]
    rep_range      = 0.5
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
    fig = plt.figure(figsize=(14, 7))
    subplot = fig.add_subplot(121)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')

    
    obstacles = [[2,2],[2.5,2],[1.75,2.25],[2.75,2.25],[2.25,2.5]]
    obstacles = filling(obstacles)
    
    APF1 = APF(start          = (0,0), 
               goal           = (5,5),
               obstacle_List  = obstacles,
               k_att          = 1.0, 
               k_rep          = 100.0,
               rep_range      = 0.5, 
               step_size      = 0.1,
               max_iters      = 500,
               goal_threshold = 1.0)
    
    k = APF1.path_plan()
    if k != None:
        k.path_plan()
    
#   fig = plt.figure(figsize=(7, 7))
    subplot = fig.add_subplot(122)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    
    for ob_pos in obstacles:
        circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius = rep_range, alpha=0.3)
        subplot.plot(ob_pos[0], ob_pos[1], 'o')
        subplot.add_patch(circle)
    
    if k == None:
        subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    else:
        subplot.plot([p[0] for p in k.path], [p[1] for p in k.path], linestyle = '-', marker = 'o')
    plt.show()
