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
    
    @classmethod
    def Tuple_init(cls, point: tuple):
        return cls(point[0], point[1])

    def property_setting(self):
        '''
            设置向量的长度，单位向量
        '''
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
        if self.length > 0:
            self.Unit_Vec = [self.deltaX / self.length, self.deltaY / self.length]
        else:
            self.Unit_Vec = None
            
    def vector_in_tuple(self)-> tuple:
        return (self.deltaX, self.deltaY)

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
    
    @staticmethod
    def distance_points(p1: tuple,p2: tuple)-> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def distance_point_line(p: tuple, line_p1: tuple, line_p2: tuple)-> float:
        vec1 = np.array(line_p1) - np.array(p)
        vec2 = np.array(line_p2) - np.array(p)
        vecP = Vector2d(line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
        distance = np.abs(np.cross(vec1,vec2)) / vecP.length
        return distance

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
#! --------------------------APF---------------------------------
    def U_att(self, pos: Vector2d)-> float:
        return 0.5 * self.k_att * ((pos - self.goal).length) ** 2

    def U_rep(self, pos: Vector2d)-> float:
        all_U = 0
        for ob in self.V_obstacle_list:
            rep_F = pos - ob
            if (rep_F.length <= self.rep_range):
                all_U += 0.5 * self.k_rep * (1 / (rep_F.length) - (1 / self.rep_range)) ** 2
        return all_U
    
    def U_edge(self, line_p1: tuple, line_p2: tuple)-> float:
        all_U = 0
        for ob in self.V_obstacle_list:
            distance = Vector2d.distance_point_line(ob.vector_in_tuple(), line_p1, line_p2)
            if (distance <= self.rep_range):
                try:
                    all_U += 0.5 * self.k_rep * (1 / (distance) - (1 / self.rep_range)) ** 2
                except ZeroDivisionError:
                    all_U = float('inf')
        return all_U

    def attractive_F(self, source: Vector2d)-> Vector2d:
        '''
            U_att(cur) = 1/2 * α * |cur - goal|^2 
            F_att(cur) = - ▽U_att(cur) = - α * (cur - goal)
        '''
        att = (source - self.current_pos) * self.k_att
        return att

    def repulsion_F(self)-> Vector2d:
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
#! --------------------------VNS---------------------------------
    def dividePath(self):
        '''generate the self.subgoals, which is uniformly spaced from the self.path[len(self.path) - self.length : ]
           subgoals 是倒过来的 [s5,s4,s3,s2,s1]
        '''
        self.subgoals.clear()
        if len(self.path) < self.length:
            self.subgoals.append(self.path[0])
            self.path.remove(self.path[0])
        else:
            INTERVAL = int(self.length / self.num_SG)
            for index in range(self.num_SG):
                SG_index = -1 - (index + 1) * INTERVAL
                self.subgoals.append(self.path[SG_index])
                
                if index == self.num_SG - 1:
                    print(SG_index - INTERVAL + len(self.path))
                    if SG_index + len(self.path) < INTERVAL:
                        self.current_pos = Vector2d.Tuple_init(self.path[0])
                        self.path = self.path[:1]
                    else:
                        self.current_pos = Vector2d.Tuple_init(self.path[SG_index - INTERVAL + len(self.path)])
                        self.path = self.path[0 : SG_index - INTERVAL + len(self.path) + 1]  

    def VNS(self, neighbourhood_names: list)-> None:
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
        
    def objective_func(self, subgoals)-> float:
        '''evaluate the subgoals in three aspects

            Args:
                subgoals (list or tuple): subgoals

            Returns:
                float: evaluation
        '''
        U_ALL  = sum([(self.U_att(Vector2d.Tuple_init(x)) - self.U_rep(Vector2d.Tuple_init(x))) for x in subgoals])
        
        D_PATH = sum([Vector2d.distance_points(self.current_pos.vector_in_tuple(), subgoals[-1])] + 
                     [Vector2d.distance_points(subgoals[i - 1], subgoals[i]) for i in range(1, len(subgoals))] + 
                     [Vector2d.distance_points(self.goal.vector_in_tuple(), subgoals[0])])
        
        U_EDGE = sum([self.U_edge(subgoals[i], subgoals[i - 1])  for i in range(1, len(subgoals))])
        a = 1
        b = 1
        c = 1
        return a * U_ALL - b * D_PATH - c * U_EDGE
    
    def path_plan(self):
        while (self.cur_iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threshold):
            
            if len(self.subgoals) != 0:
                f_vec = self.attractive_F(self.subgoals[-1]) + self.repulsion_F()
                self.current_pos += Vector2d.Tuple_init(f_vec.Unit_Vec)  * self.step_size 
                if (len(self.path) >= 3 and (Vector2d.Tuple_init(self.path[-2]) - self.current_pos).length < self.step_size):
                    self.dividePath()
                    self.VNS(self.N_order)
                else:
                    if (self.current_pos - Vector2d.Tuple_init(self.subgoals[-1])).length <= self.step_size:
                        self.subgoals.remove(self.subgoals[-1])
                    else:
                        self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
            else:
                f_vec = self.attractive_F(self.goal) + self.repulsion_F()
                self.current_pos += Vector2d.Tuple_init(f_vec.Unit_Vec) * self.step_size
                if (len(self.path) >= 3 and (Vector2d.Tuple_init(self.path[-2]) - self.current_pos).length < self.step_size):
                    self.dividePath()
                    self.VNS(self.N_order)
                else:
                    self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
                    
            self.cur_iters += 1
        if (self.current_pos - self.goal).length <= self.goal_threshold:
            self.is_path_plan_success = True
#! --------------------------neighbourhood---------------------------------  
    def neighbourhood_random(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        result = set()
        for _ in range(8):
            neighbour = list()
            for sg in cur_SGs:
                ran_degree = random.randint(0, 360)
                cos = math.cos(math.radians(ran_degree))
                sin = math.sin(math.radians(ran_degree))
                # vec = Vector2d(cos, sin) * self.step_size
                neighbour.append((sg[0] + cos * self.step_size, sg[1] + sin * self.step_size))
            result.add(tuple(neighbour))
        return result
    
    def neighbourhood_up(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_dowm(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_left(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_right(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_random_eight(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_obs_free(self, cur_SGs : list)-> set:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                set: the set of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}
    
    def neighbourhood_optimize_edge(self, cur_SGs : list)-> set:
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
            # *neighbourhood name domain = {"neighbourhood_up", "neighbourhood_dowm", "neighbourhood_left", "neighbourhood_right",
            # *                             "neighbourhood_random", "neighbourhood_random_eight", "neighbourhood_obs_free", "neighbourhood_optimize_edge"}

            Args:
                name (str): name of the neighbourhood we want 
                cur_SGs (list): current subgoals
            Returns:
                set: set of all of this neighbourhood solutions
        '''
        if name == "neighbourhood_random":
            return self.neighbourhood_random(cur_SGs)
        elif name == "neighbourhood_up":
            return self.neighbourhood_up(cur_SGs)
        elif name == "neighbourhood_dowm":
            return self.neighbourhood_dowm(cur_SGs)
        elif name == "neighbourhood_left":
            return self.neighbourhood_left(cur_SGs)
        elif name == "neighbourhood_right":
            return self.neighbourhood_right(cur_SGs)
        elif name == "neighbourhood_random_eight":
            return self.neighbourhood_random_eight(cur_SGs)
        elif name == "neighbourhood_obs_free":
            return self.neighbourhood_obs_free(cur_SGs)
        elif name == "neighbourhood_optimize_edge":
            return self.neighbourhood_optimize_edge(cur_SGs)
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
    APF1.path = [
                [0.1414213562373095, 0.1414213562373095],
                [0.282842712474619, 0.282842712474619], #*
                [0.4242640687119285, 0.4242640687119285],
                [0.565685424949238, 0.565685424949238],
                [0.7071067811865475, 0.7071067811865475],
                [0.848528137423857, 0.848528137423857],  #! 1
                [0.9899494936611666, 0.9899494936611666],
                [1.131370849898476, 1.131370849898476], 
                [1.2727922061357855, 1.2727922061357855], 
                [1.414213562373095, 1.414213562373095], #! 2
                [1.5556349186104044, 1.5556349186104044],
                [1.6970562748477138, 1.6970562748477138],
                [1.8384776310850233, 1.8384776310850233],
                [1.9798989873223327, 1.9798989873223327], #! 3
                [2.1213203435596424, 2.1213203435596424],
                [2.262741699796952, 2.262741699796952],
                [2.4041630560342617, 2.4041630560342617],
                [2.5455844122715714, 2.5455844122715714], #! 4
                [3,3],
                [4,4],
                [5,5],
                [6,6]
                ]
    print(len(APF1.path))    
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
