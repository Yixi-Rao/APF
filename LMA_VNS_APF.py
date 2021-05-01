import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import random
import functools
import numpy as np

class Vector2d():
    def __init__(self, x, y):
        self.deltaX = x        # 表示向量时，是x方向上的变化量。表示点时是起点是原点的向量
        self.deltaY = y
        self.length = -1       # 向量长度
        self.Unit_Vec = [0, 0]   # 此向量的单位向量
        self.property_setting()

    def property_setting(self):
        '''
            设置向量的长度，单位向量
        '''
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
        if self.length > 0:
            self.Unit_Vec = [self.deltaX /
                             self.length, self.deltaY / self.length]
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
        return 'Vector X:{}, Y:{}, length:{}, Unit_Vec:{}'.format(self.deltaX, self.deltaY, self.length, self.Unit_Vec)

class APF_VNS():
    def __init__(self,
                 start:     tuple, goal:            tuple, obstacle_List: list,
                 k_att:     float, k_rep:           float,
                 rep_range: float, step_size:       float, max_iters:     int, goal_threshold: float,
                 length:     int,  number_subgoals: int,
                 N_order: list):
        
        self.start           = Vector2d(start[0], start[1])                     # 起点vector
        self.goal            = Vector2d(goal[0], goal[1])                       # 终点vector
        self.V_obstacle_list = [Vector2d(OB[0], OB[1]) for OB in obstacle_List] # 障碍物列表，每个元素为Vector2d对象
        self.k_att           = k_att           # 引力系数
        self.k_rep           = k_rep           # 斥力系数
        self.rep_range       = rep_range       # 斥力作用范围
        self.step_size       = step_size       # 步长
        self.max_iters       = max_iters       # 最大迭代次数
        self.goal_threshold  = goal_threshold  # 离目标点小于此值即认为到达目标点

        self.cur_iters   = 0                            # 当前迭代数
        self.current_pos = Vector2d(start[0], start[1]) # 当前位置: vector
        self.path        = list()                       # 规划路径: list(tuple)
        self.length      = length                       # 规定提取子目标的path的长度
        self.num_SG      = number_subgoals              # 子目标个数
        self.subgoals    = list()                       # 子目标: list(tuple)
        self.N_order     = N_order                      # *领域的顺序: domain = {"neihgbourhood_up", "neihgbourhood_dowm", "neihgbourhood_left", "neihgbourhood_right",
                                                        # *                     "neihgbourhood_random", "neihgbourhood_random_eight", "neihgbourhood_obs_free", "neihgbourhood_optimize_edge"}
        
        self.is_path_plan_success = False               # 是否陷入不可达或局部最小值

    def U_att(self, pos):
        return 0.5 * self.k_att * ((pos - self.goal).length) ** 2

    def U_rep(self, pos):
        all_U = 0
        for ob in self.V_obstacle_list:
            rep_F = self.current_pos - ob
            if (rep_F.length <= self.rep_range):
                all_U += 0.5 * self.k_rep * (1 / (rep_F.length) - (1 / self.rep_range)) ** 2
        return all_U

    def attractive_F(self, source):
        '''
            U_att(cur) = 1/2 * α * |cur - goal|^2 
            F_att(cur) = - ▽U_att(cur) = - α * (cur - goal)
        '''
        att = (source - self.current_pos) * self.k_att
        return att

    def repulsion_F(self):
        """
            U_rep(cur) = 1/2 * β * (1 / (|cur - goal|) - 1 / rr)^2
            F_rep(cur) = β * (1 / (|cur - goal|) - 1 / rr) * 1/(|cur - goal|^2) * ▽(|cur - goal|)
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for ob in self.V_obstacle_list:
            rep_F = self.current_pos - ob  # 矢量斥力方向
            if (rep_F.length <= self.rep_range):  # 在斥力影响范围
                rep += Vector2d(rep_F.Unit_Vec[0], rep_F.Unit_Vec[1]) * self.k_rep * (1.0 / rep_F.length - 1.0 / self.rep_range) / (rep_F.length ** 2)
        return rep

    def dividePath(self):
        '''generate the self.subgoals, which is uniformly spaced from the self.path[len(self.path) - self.length : ]
        '''
        self.subgoals.clear()
        if len(self.path) < self.length:
            self.subgoals.append(self.path[0])
            self.path.remove(self.path[0])
        else:
            interval = int(self.length / self.num_SG)
            for index in range(self.num_SG):
                self.subgoals.append(self.path[-1 - (index + 1) * interval])
                if index == self.num_SG - 1:
                    self.path = self.path[0 : -1 - (index + 1) * interval + len(self.path)]
                    self.current_pos = self.path[-1 - (index + 2) * interval + len(self.path)]
                    # TODO: index out of bound
                    
    def VNS(self, neighbourhood_names: list):
        '''find the subgoals that can help the agent escape from the LM

            Args:
                neighbourhood_names (list): list of the neighbourhood function name e.g. ["neihgbour_up","neihgbour_left"...]
        '''
        t = 1
        k = 1
        cur_SGs = self.subgoals
        
        while k != len(neighbourhood_names) and t != 50:
            shaked_SGs = self.shuffing(cur_SGs)
            local_SGs  = self.bestImprovement(shaked_SGs, self.neighbourhood_selector(neighbourhood_names[k - 1], shaked_SGs))
            if self.objective_func(local_SGs) < self.objective_func(cur_SGs):
                cur_SGs = local_SGs
                k = 1
            else:
                k = k + 1
            t = t + 1
    
    def bestImprovement(self, shaked_SGs: list, N: set)-> list:
        '''a local serach

            Args:
                shaked_SGs (list): a shaked subgoal list
                N (set): set of neighbour subgoals

            Returns:
                list: best neighbour in N
        '''
        temp  = shaked_SGs.copy()
        t_max = 1
        while True and t_max < 20:
            local_SGs = temp
            min_val   = min(map(self.objective_func, N))
            temp      = list(filter(lambda x: self.objective_func(x) == min_val, N))[0]
            t_max += 1
            if min_val >= self.objective_func(local_SGs):
                return temp
    
    def shuffing(self, cur_SGs: list)-> list:
        '''randomly change exact one subgoal in the cur_SGs by randomly select a direction

            Args:
                cur_SGs (list): current subgoals

            Returns:
                [list]: new shaked subgoals
        '''
        ran_index = random.randint(0, len(cur_SGs) - 1)
        result    = cur_SGs.copy()
        while True:
            ran_degree = random.randint(0, 360)
            cos = math.cos(math.radians(ran_degree))
            sin = math.sin(math.radians(ran_degree))
            vec = Vector2d(cos, sin) * self.step_size
            new_pos = (self.subgoals[ran_index][0] + vec.Unit_Vec[0], self.subgoals[ran_index][1] + vec.Unit_Vec[1])
            if new_pos not in self.V_obstacle_list:
                result[ran_index] = new_pos
                return result
        
    def objective_func(self, subgoals: list)-> float:
        '''evaluate the subgoals in three aspects

            Args:
                subgoals (list): subgoals

            Returns:
                float: evaluation
        '''
        return 0.0
    
    def path_plan(self):
        while (self.cur_iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threshold):
            
            if len(self.subgoals) != 0:
                f_vec = self.attractive_F(self.subgoals[0]) + self.repulsion_F()
                self.current_pos += Vector2d(f_vec.Unit_Vec[0], f_vec.Unit_Vec[1]) * self.step_size
                if (len(self.path) >= 3 and (Vector2d(self.path[-2][0], self.path[-2][1]) - self.current_pos).length < self.step_size):
                    self.dividePath()
                    self.VNS(self.N_order)
                else:
                    if (self.current_pos - self.subgoals[0]).length <= self.step_size:
                        self.subgoals.remove(self.subgoals[0])
                    else:
                        self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
            else:
                f_vec = self.attractive_F(self.goal) + self.repulsion_F()
                self.current_pos += Vector2d(f_vec.Unit_Vec[0], f_vec.Unit_Vec[1]) * self.step_size
                if (len(self.path) >= 3 and (Vector2d(self.path[-2][0], self.path[-2][1]) - self.current_pos).length < self.step_size):
                    self.dividePath()
                    self.VNS(self.N_order)
                else:
                    self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
                    
            self.cur_iters += 1
        if (self.current_pos - self.goal).length <= self.goal_threshold:
            self.is_path_plan_success = True
   
    def neighbourhood_random(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_selector(self, name: str, cur_SGs : list)-> set:
        '''select the right neighbourhood by using the name
            # *neighbourhood name domain = {"neihgbourhood_up", "neihgbourhood_dowm", "neihgbourhood_left", "neihgbourhood_right",
            # *                             "neihgbourhood_random", "neihgbourhood_random_eight", "neihgbourhood_obs_free", "neihgbourhood_optimize_edge"}

            Args:
                name (str): name of the neighbourhood we want 
                cur_SGs (list): current subgoals
            Returns:
                set: set of all of this neighbourhood solutions
        '''
        if name == "neihgbourhood_random":
            return self.neighbourhood_random(cur_SGs)
        elif name == "neihgbourhood_up":
            pass
        elif name == "neihgbourhood_dowm":
            pass
        elif name == "neihgbourhood_left":
            pass
        elif name == "neihgbourhood_right":
            pass
        elif name == "neihgbourhood_random_eight":
            pass
        elif name == "neihgbourhood_obs_free":
            pass
        elif name == "neihgbourhood_optimize_edge":
            pass
        else:
            raise ValueError("name: " + name + " - this Neighbourhood does not exist!!!")
    

if __name__ == '__main__':
    start = (0, 0)
    goal  = (5, 5)
    
    obstacle_List1 = [[3, 3], [2.5, 3], [2, 3], [3, 2], [3, 2.5]]  # 低效率的环形障碍物不可达# 目标障碍物同一直线不可达问题
    obstacle_List2 = [[3, 3]]
    
    k_att          = 1
    k_rep          = 20
    rep_range      = 0.6
    step_size      = 0.2
    max_iters      = 1000
    goal_threshold = 1.0
    length         = 18
    num_sub        = 4
    
    neighbour_name = ["neihgbourhood_up", "neihgbourhood_dowm", "neihgbourhood_left", "neihgbourhood_right"]
    
    APF1 = APF_VNS(start, goal, obstacle_List1, k_att, k_rep, rep_range, step_size, max_iters, goal_threshold, length, num_sub, neighbour_name)
    # APF1.path_plan()
    
    # fig = plt.figure(figsize=(7, 7))
    # subplot = fig.add_subplot(111)
    # subplot.set_xlabel('X')
    # subplot.set_ylabel('Y')
    # subplot.plot(start[0], start[1], 'X')
    
    # circle_goal = Circle(xy=(goal[0], goal[1]), radius=goal_threshold, alpha=0.9)
    # subplot.plot(goal[0], goal[1], 'X')
    # subplot.add_patch(circle_goal)
    
    # for ob_pos in obstacle_List1:
    #     circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.3)
    #     subplot.plot(ob_pos[0], ob_pos[1], 'o')
    #     subplot.add_patch(circle)
        
    # subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle='-', marker='o')
    # plt.show()
