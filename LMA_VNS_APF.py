import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import random
import queue

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
    
    @staticmethod
    def Is_PointOnLine(p: tuple, line_p1: tuple, line_p2: tuple)-> bool:
        return Vector2d.distance_points(p, line_p1) + Vector2d.distance_points(p, line_p2) == Vector2d.distance_points(line_p1, line_p2)

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
        self.stuck_paths = list()
        self.all_subgoals = list()
        self.first_stuck = list()
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
            
            if not Vector2d.Is_PointOnLine(ob.vector_in_tuple(), line_p1, line_p2):
                continue
            elif distance == 0:
                all_U = float('inf')
            else:
                all_U += 0.5 * self.k_rep * (1 / (distance) - (1 / self.rep_range)) ** 2
            
        return all_U

    def attractive_F(self, source: Vector2d)-> Vector2d:
        '''
            U_att(cur) = 1/2 * α * |cur - goal|^2 
            F_att(cur) = - ▽U_att(cur) = α * (cur - goal)
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
                    if SG_index + len(self.path) < INTERVAL:
                        self.current_pos = Vector2d.Tuple_init(self.path[0])
                        self.path = self.path[:1]
                    else:
                        self.current_pos = Vector2d.Tuple_init(self.path[SG_index - INTERVAL + len(self.path)])
                        self.path = self.path[0 : SG_index - INTERVAL + len(self.path) + 1]
                        
    def dividePath2(self):
        '''generate the self.subgoals, which is uniformly spaced from the self.path[len(self.path) - self.length : ]
           subgoals 是倒过来的 [s5,s4,s3,s2,s1]
        '''
        self.subgoals.clear()
        
        if len(self.path) < self.length:
            self.subgoals.append(self.path[0])
            self.path.remove(self.path[0])
        else:
            INTERVAL = int(self.length / self.num_SG)
            SG_index = -1 - INTERVAL
            for index in range(self.num_SG):
                while True and len(self.subgoals) != 0:
                    if (SG_index + len(self.path) < 0):
                        SG_index = 0
                        break
                    protential_sg = self.path[SG_index]
                    if Vector2d.distance_points(protential_sg, self.subgoals[0]) < self.step_size * (INTERVAL / 2):
                        SG_index = SG_index - INTERVAL
                    else:
                        break
                
                self.subgoals.append(self.path[SG_index])
                SG_index = SG_index - INTERVAL
                
                if index == self.num_SG - 1:
                    if SG_index + len(self.path) < INTERVAL:
                        self.current_pos = Vector2d.Tuple_init(self.path[0])
                        self.path = self.path[:1]
                    else:
                        self.current_pos = Vector2d.Tuple_init(self.path[SG_index - INTERVAL + len(self.path)])
                        self.path = self.path[0 : SG_index - INTERVAL + len(self.path) + 1] 

    def VNS(self, neighbourhood_names: list):
        '''find the subgoals that can help the agent escape from the LM

            Args:
                neighbourhood_names (list): list of the neighbourhood function name e.g. ["neihgbour_up","neihgbour_left"...]
        '''
        t = 1
        k = 1
        cur_SGs = self.subgoals
        
        while k != len(neighbourhood_names) and t != 70:
            shaked_SGs = self.shuffing(cur_SGs)
            local_SGs  = self.bestImprovement(shaked_SGs, self.neighbourhood_selector(neighbourhood_names[k - 1], shaked_SGs))
            if self.objective_func(local_SGs) > self.objective_func(cur_SGs):
                cur_SGs = local_SGs
                k = 1
            else:
                k = k + 1
            t = t + 1
        self.subgoals = cur_SGs
        self.all_subgoals.append(self.subgoals.copy()) #! delete safe
        
    def bestImprovement(self, shaked_SGs: list, N: list)-> list:
        '''a local serach

            Args:
                shaked_SGs (list): a shaked subgoal list
                N (list): list of neighbour subgoals

            Returns:
                list: best neighbour in N
        '''
        temp  = shaked_SGs.copy()
        t_max = 1
        while True and t_max < 20:
            local_SGs = temp
            max_val   = max(map(self.objective_func, N))
            temp      = list(filter(lambda x: self.objective_func(x) == max_val, N))[0]
            t_max += 1
            if max_val <= self.objective_func(local_SGs):
                return local_SGs
    
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
            
            new_pos = (self.subgoals[ran_index][0] + cos * self.step_size, self.subgoals[ran_index][1] + sin * self.step_size) 
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
        b = 20
        c = 10
        return a * U_ALL - b * D_PATH - c * U_EDGE
    
    def path_plan(self):
        first = True
        while (self.cur_iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threshold):
            
            if len(self.subgoals) != 0:
                f_vec = self.attractive_F(Vector2d.Tuple_init(self.subgoals[-1])) + self.repulsion_F()
                self.current_pos += Vector2d.Tuple_init(f_vec.Unit_Vec)  * self.step_size 
                if (len(self.path) >= 3 and (Vector2d.Tuple_init(self.path[-2]) - self.current_pos).length < self.step_size / 100):
                    self.dividePath2()
                    self.VNS(self.N_order)
                else:
                    if (self.current_pos - Vector2d.Tuple_init(self.subgoals[-1])).length <= self.step_size:
                        self.subgoals.remove(self.subgoals[-1])
                    else:
                        self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
            else:
                f_vec = self.attractive_F(self.goal) + self.repulsion_F()
                self.current_pos += Vector2d.Tuple_init(f_vec.Unit_Vec) * self.step_size
                if (len(self.path) >= 3 and (Vector2d.Tuple_init(self.path[-2]) - self.current_pos).length < self.step_size / 100):
                    self.dividePath2()
                    if first:
                        first = False
                        self.first_stuck = (self.subgoals.copy())
                    self.VNS(self.N_order)
                    
                else:
                    self.path.append((self.current_pos.deltaX, self.current_pos.deltaY))
                    
            self.cur_iters += 1
        if (self.current_pos - self.goal).length <= self.goal_threshold:
            self.is_path_plan_success = True
#! --------------------------neighbourhood---------------------------------  
    def neighbourhood_random(self, cur_SGs : list)-> list:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        for _ in range(8):
            neighbour = list()
            for sg in cur_SGs:
                ran_degree = random.randint(0, 360)
                cos = math.cos(math.radians(ran_degree))
                sin = math.sin(math.radians(ran_degree))
                neighbour.append((sg[0] + cos * self.step_size * 0.8, sg[1] + sin * self.step_size * 0.8))
            result.append(neighbour)
        return result
    
    def neighbourhood_random_eight(self, cur_SGs : list)-> list:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        for _ in range(8):
            neighbour = list()
            for sg in cur_SGs:
                ran_degree = random.randint(1, 8) * 45
                cos = math.cos(math.radians(ran_degree))
                sin = math.sin(math.radians(ran_degree))
                neighbour.append((sg[0] + cos * self.step_size * 0.8, sg[1] + sin * self.step_size * 0.8))
            result.append(neighbour)
        return result
    
    def neighbourhood_up(self, cur_SGs : list)-> list:
        '''change the direction up

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        interval = self.step_size / 2
        for i in range(8):
            neighbour = list()
            for sg in cur_SGs:
                neighbour.append((sg[0], sg[1] + interval * (i + 1)))
            result.append(neighbour)
        return result

    def neighbourhood_down(self, cur_SGs : list)-> list:
        '''change the direction down

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        interval = self.step_size / 2
        for i in range(8):
            neighbour = list()
            for sg in cur_SGs:
                neighbour.append((sg[0], sg[1] - interval * (i + 1)))
            result.append(neighbour)
        return result

    def neighbourhood_left(self, cur_SGs : list)-> list:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        interval = self.step_size / 2
        for i in range(8):
            neighbour = list()
            for sg in cur_SGs:
                neighbour.append((sg[0] - interval * (i + 1), sg[1]))
            result.append(neighbour)
        return result

    def neighbourhood_right(self, cur_SGs : list)-> list:
        '''randomly change the direction in 360 degrss

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        interval = self.step_size / 2
        for i in range(8):
            neighbour = list()
            for sg in cur_SGs:
                neighbour.append((sg[0] + interval * (i + 1), sg[1]))
            result.append(neighbour)
        return result
    
    def neighbourhood_U_D_L_R(self, cur_SGs : list)-> list:
        '''up down left right ramdomly

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result    = list()
        interval  = self.step_size / 2
        direction = random.randint(1, 4)
        
        if direction == 1:
            for i in range(8):
                neighbour = list()
                for sg in cur_SGs:
                    neighbour.append((sg[0], sg[1] + interval * (i + 1)))
                result.append(neighbour)
            return result
        elif direction == 2:
            for i in range(8):
                neighbour = list()
                for sg in cur_SGs:
                    neighbour.append((sg[0], sg[1] - interval * (i + 1)))
                result.append(neighbour)
            return result
        elif direction == 3:
            result = list()
            interval = self.step_size / 2
            for i in range(8):
                neighbour = list()
                for sg in cur_SGs:
                    neighbour.append((sg[0] - interval * (i + 1), sg[1]))
                result.append(neighbour)
            return result
        else:
            for i in range(8):
                neighbour = list()
                for sg in cur_SGs:
                    neighbour.append((sg[0] + interval * (i + 1), sg[1]))
                result.append(neighbour)
            return result
        
    def neighbourhood_obs_free(self, cur_SGs: list)-> list:
        '''randomly change the direction avoiding obstacles

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        for _ in range(8):
            neighbour = list()
            for sg in cur_SGs:
                ava_queue = queue.PriorityQueue()
                interval  = self.step_size
                for run in range(35):
                    #! magnify the step_size to increase the chance to find the ideal sg
                    if run == 10:
                        interval = self.step_size * 1.5
                    elif run == 20:
                        interval = self.step_size * 2
                    elif run == 30:
                        interval = self.step_size * 2.5
                    #! generate the new subgoal
                    ran_degree = random.randint(0, 360)
                    cos        = math.cos(math.radians(ran_degree))
                    sin        = math.sin(math.radians(ran_degree))
                    new_sg     = (sg[0] + cos * interval, sg[1] + sin * interval)
                    num_in_obs = self.Count_obstacles(new_sg)
                    #! find an obstacle-free new subgoal    
                    if num_in_obs == 0:
                        neighbour.append(new_sg)
                        break
                    else:
                        ava_queue.put((num_in_obs, new_sg))
                    #! even in the last run, we cannot find a new_sg which is not in the obstacle areas, so we choose the new_sg that is contained in minimal number of obstacle areas   
                    if run == 34:
                        new_sg = ava_queue.get()[1]
                        neighbour.append(new_sg)
                        
            result.append(neighbour)
        return result    
            
    def neighbourhood_optimize_edge(self, cur_SGs : list)-> list:
        '''生成这个领域的可能的8个解的list，此领域为智能道路优化领域，这个领域的解会尽量减少U_edge的值。
           具体：每一个subgoal都会360°位移，新的位移的点与他相邻的subgoals连线计算的U_edge值会减小

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        # TODO: neighbourhood
        return {cur_SGs}

    def neighbourhood_highest_poss(self, cur_SGs : list)-> list:
        '''change the direction where is farest from the nearest obstacle

            Args:
                cur_SGs (list): current positon

            Returns:
                list: the list of neighbour solution
        '''
        result = list()
        for _ in range(8):
            neighbour = list()
            for sg in cur_SGs:
                nearest_ob = None
                min_d = 100000000.0
                for ob in self.V_obstacle_list:
                    d = Vector2d(sg[0], sg[1]) - ob
                    if (d.length <= min_d ):  # 在斥力影响范围
                        min_d = d.length
                        nearest_ob = ob
                min_vec = [0.0,1.0]
                max_d = 0.0
                for degree in range(0,360):
                    cos = math.cos(math.radians(degree))
                    sin = math.sin(math.radians(degree))
                    d = Vector2d(sg[0] + cos * self.step_size, sg[1] + sin * self.step_size) - nearest_ob
                    if d.length > max_d:
                        min_vec = [cos,sin]
                neighbour.append((sg[0] + min_vec[0] * self.step_size, sg[1] + min_vec[1] * self.step_size))
            result.append(neighbour)
        return result

    def neighbourhood_selector(self, name: str, cur_SGs : list)-> list:
        '''select the right neighbourhood by using the name
            #* neighbourhood name domain = {"neighbourhood_up", "neighbourhood_dowm", "neighbourhood_left", "neighbourhood_right",
            #*                              "neighbourhood_random", "neighbourhood_random_eight", "neighbourhood_obs_free", "neighbourhood_optimize_edge"}

            Args:
                name (str): name of the neighbourhood we want 
                cur_SGs (list): current subgoals
            Returns:
                list: list of all of this neighbourhood solutions
        '''
        if name == "neighbourhood_random":
            return self.neighbourhood_random(cur_SGs)
        elif name == "neighbourhood_up":
            return self.neighbourhood_up(cur_SGs)
        elif name == "neighbourhood_down":
            return self.neighbourhood_down(cur_SGs)
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
        elif name == "neighbourhood_highest_poss":
            return self.neighbourhood_highest_poss(cur_SGs)
        elif name == "neighbourhood_U_D_L_R":
            return self.neighbourhood_U_D_L_R(cur_SGs)
        else:
            raise ValueError("name: " + name + " - this Neighbourhood does not exist!!!")

    def Count_obstacles(self, pos: tuple):
        count = 0
        for ob in self.V_obstacle_list:
            d = Vector2d.distance_points(pos, ob.vector_in_tuple())
            if d < self.rep_range:
                count += 1   
        return count
    
#! --------------------------main---------------------------------
if __name__ == '__main__':
    start = (0, 0)
    goal  = (10, 10)
    
    obstacle_List1 = [[6, 6]]
    obstacle_List2 = [[6, 6], [5.75, 6], [5.5, 6], [6, 5.75], [6, 5.5]]
    obstacle_List3 = [[6, 6], [5.75, 6], [6, 5.75]]
    obstacle_List4 = [[6, 6], [5.75, 6], [5.5, 6], [6, 5.75], [6, 5.5],[6, 5],
                      [3.5,4.6], [1.55,1.48], [6,3.83],[6.35,7.45],[4.7,6.9],
                      [6,2.05],[1.87,7.07],[8.97,4.99], [3, 7], [3.25, 7], [3.5, 7], [3, 6.75], [3, 6.5]]
    
    k_att          = 1
    k_rep          = 20
    rep_range      = 0.6
    step_size      = 0.2
    max_iters      = 1000
    goal_threshold = 1.0
    length         = 18
    num_sub        = 4
    
    #* four direction neighbourhoods list
    # neighbour_name = ["neighbourhood_left", "neighbourhood_right", "neighbourhood_down", "neighbourhood_up"]
    
    #* random neighbourhoods list 
    # neighbour_name = ["neighbourhood_random_eight", "neighbourhood_random"]
    
    #* four direction + random neighbourhoods list
    # neighbour_name = ["neighbourhood_up", "neighbourhood_right", "neighbourhood_left", "neighbourhood_down", "neighbourhood_random_eight", "neighbourhood_random"]
    
    #* path optimization neighbourhoods list 
    # neighbour_name = ["neighbourhood_obs_free","neighbourhood_highest_poss"]
   
    #* all neighbourhoods list 
    # neighbour_name = ["neighbourhood_up", "neighbourhood_down", "neighbourhood_left", "neighbourhood_right", "neighbourhood_random_eight", "neighbourhood_random","neighbourhood_obs_free","neighbourhood_highest_poss"]
     
    #* UDLR neighbourhoods list
    neighbour_name = ["neighbourhood_U_D_L_R", "neighbourhood_random", "neighbourhood_obs_free", "neighbourhood_highest_poss"]
    
    APF1 = APF_VNS(start, goal, obstacle_List4, k_att, k_rep, rep_range, step_size, max_iters, goal_threshold, length, num_sub, neighbour_name)
    
    APF1.path_plan()
    #! figure configuration 
    fig      = plt.figure(figsize=(24, 12))
    subplot  = fig.add_subplot(1,2,1) # 子图1：显示path和最后一个subgoals
    subplot2 = fig.add_subplot(1,2,2) # 子图2：显示所有探索过的subgoals
    
    #! subplot1 configuration 
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
    subplot.set_title('Path graph')
    subplot.plot(start[0], start[1], marker='h', color='r')
    
    circle_goal = Circle(xy=(goal[0], goal[1]), radius = goal_threshold, alpha=0.1, color = 'r', )
    subplot.plot(goal[0], goal[1], label = "goal", marker='*', color='r', markersize =14)
    subplot.add_patch(circle_goal)
    
    #! subplot2 configuration 
    subplot2.set_xlabel('X')
    subplot2.set_ylabel('Y')
    subplot2.set_title('subgoals graph')
    
    #! all subplots obstacles configuration 
    token = True
    for ob_pos in obstacle_List4:
        if token:
            token = False
            circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.3, color = 'black')
            subplot.plot(ob_pos[0], ob_pos[1], 's', label = "obstacle", markersize = 10, color = '#4169E1')
            subplot.add_patch(circle)
            
            circle2 = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.1, color = 'black')
            subplot2.plot(ob_pos[0], ob_pos[1], 's', label = "obstacle", markersize = 10, color = '#4169E1',alpha=1)
            subplot2.add_patch(circle2)
        else:
            circle = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.3, color = 'black')
            subplot.plot(ob_pos[0], ob_pos[1], 's', markersize = 10, color = '#4169E1')
            subplot.add_patch(circle)
            
            circle2 = Circle(xy=(ob_pos[0], ob_pos[1]), radius=rep_range, alpha=0.1, color = 'black')
            subplot2.plot(ob_pos[0], ob_pos[1], 's', markersize = 10, color = '#4169E1',alpha=1)
            subplot2.add_patch(circle2)
    
    #! subplot1 lines configuration    
    subplot.plot([p[0] for p in APF1.path], [p[1] for p in APF1.path], linestyle='-', marker='o', color = 'g', label = "path")
    subplot.plot([p[0] for p in APF1.all_subgoals[-1]], [p[1] for p in APF1.all_subgoals[-1]],markersize = 20 , marker='*', color = 'r',alpha=0.3, label = "final subgoals")
    subplot.plot([p[0] for p in APF1.first_stuck], [p[1] for p in APF1.first_stuck], linestyle=':', markersize = 15,marker='*', color = 'g',label = "first stuck subgoals", alpha = 0.3)
    
    #! subplot2 lines configuration  
    color = ['b','r','y','c','k','m','y','b','r','y','c','k','m','k','b','r','y','c','k','m','c',
             'b','r','y','c','k','m','y','b','r','y','c','k','m','k','b','r','y','c','k','m','c',
             'b','r','y','c','k','m','y','b','r','y','c','k','m','r','b','r','y','c','k','m','c',]
    
    subplot2.plot([p[0] for p in APF1.first_stuck], [p[1] for p in APF1.first_stuck], linestyle='-', markersize = 15,marker='o', color = 'r',label = "initial subgoals", alpha = 0.3)
    
    for i, path in enumerate(APF1.all_subgoals):
        subplot2.plot([p[0] for p in path], [p[1] for p in path], linestyle=':', marker='*', color = color[i],alpha=0.8)
        
    #! show   
    subplot.legend(loc = 'lower right')
    subplot2.legend(loc = 'lower right')
    plt.show()
    