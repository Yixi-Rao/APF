import math

from matplotlib import pyplot as plt
from matplotlib.patches import Circle


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
            rep_F = self.current_pos - ob # 矢量斥力方向
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
    
if __name__ == '__main__':
    # APF1 = APF(start          = (0,0), 
    #            goal           = (5,5),
    #            obstacle_List  = [[3,3]],
    #            k_att          = 1.0, 
    #            k_rep          = 100.0,
    #            rep_range      = 0.5, 
    #            step_size      = 0.1,
    #            max_iters      = 500,
    #            goal_threshold = 1.0)
    # APF1.path_plan()
    
    # fig = plt.figure(figsize=(7, 7))
    # subplot = fig.add_subplot(111)
    # subplot.set_xlabel('X')
    # subplot.set_ylabel('Y')
    # circle = Circle(xy = (2, 2.5), radius = 0.5, alpha = 0.3)
    # circle1 = Circle(xy = (1.9,1.5), radius = 0.5, alpha = 0.3)
    # subplot.plot(1.5,1.5, 'o')
    # subplot.plot(2, 2.5, 'o')
    # subplot.add_patch(circle)
    # subplot.add_patch(circle1)
    # subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle = '-', marker = 'o')
    
    # plt.show()
    
    start = (0, 0)
    goal  = (10, 10)
    obstacle_List1 = [[3, 3], [2.5, 3], [2, 3], [3, 2], [3, 2.5]] # 低效率的环形障碍物不可达
    obstacle_List2 = [[3,3]]                                      # 目标障碍物同一直线不可达问题
    obstacle_List4 = [[6, 6], [5.75, 6], [5.5, 6], [6, 5.75], [6, 5.5],[6, 5],
                      [3.5,4.6], [1.55,1.48], [6,3.83],[6.35,7.45],[4.7,6.9],
                      [6,2.05],[1.87,7.07],[8.97,4.99], [3, 7], [3.25, 7], [3.5, 7], [3, 6.75], [3, 6.5]]
    k_att          = 1
    k_rep          = 20
    rep_range      = 0.6
    step_size      = 0.2
    max_iters      = 1500
    goal_threshold = 1.0

    APF1 = APF(start, goal, obstacle_List4, k_att, k_rep, rep_range, step_size, max_iters, goal_threshold)
    APF1.path_plan()

    fig     = plt.figure(figsize=(12, 12))
    subplot = fig.add_subplot(111)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    subplot.set_title('Path graph')
    
    subplot.plot(start[0], start[1], marker='h', color='r')
    circle_goal = Circle(xy=(goal[0], goal[1]), radius = goal_threshold, alpha=0.1, color = 'r', )
    subplot.plot(goal[0], goal[1], label = "goal", marker='*', color='r', markersize =14)
    subplot.add_patch(circle_goal)
    
    token = True
    for ob_pos in obstacle_List4:
        if token:
            token = False
            circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.3, color = 'black')
            subplot.plot(ob_pos[0], ob_pos[1], 's', label = "obstacle", markersize = 10, color = '#4169E1')
            subplot.add_patch(circle)
        else:
            circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.3, color = 'black')
            subplot.plot(ob_pos[0], ob_pos[1], 's', markersize = 10, color = '#4169E1')
            subplot.add_patch(circle)
        
    subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle='-', marker='o', color = 'g', label = "path")

    plt.show()
    