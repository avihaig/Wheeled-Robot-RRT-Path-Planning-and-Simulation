import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import ones,vstack
from numpy.linalg import lstsq
from WheeledCar import WheeledCar

p_g = np.array([0.25, 0.25]) # Set <x_g, y_g> goal position
x_r = np.array([2.3, 0.25 , np.deg2rad(110)]) # Set initial pose of the robot: <x_r, y_r, phi_r>
O = np.array([[0.5, 2], [0.5, 0.5], [2, 2], [1.75, 0.3]]) # Set positions of obstacles [p_1, p_2, p_3, p_4]



# prepare the obstacle list by adding a wall between p1-p2 and p3-p4 and adding the distance requiere to stand in the allowed zone
Obstacles_for_RRT = O.tolist()
def create_line_obstacle(start_point,final_point):
    # Calculate parameters of the line equation (y = mx + c)
    x_0,y_0 = start_point[0],start_point[1]
    x_1,y_1 = final_point[0],final_point[1]

    points = [(x_0,y_0),(x_1,y_1)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    line = []
    # Generate points along the line within a certain range
    if x_1 > x_0 :
        steps = np.linspace(x_0,x_1, num = 20).tolist()
        for step in steps:
            calculate_y = ((m*step)+c)
            line.append([step,calculate_y])
    if x_1 < x_0:
        steps = np.linspace(x_1,x_0,num=20)[::-1].tolist()
        for step in steps:
            calculate_y = (m*step+c)
            line.append([step,calculate_y])
    elif x_0 == x_1 :
        if y_0 == y_1:
            line = [[x_0,y_0]]
        if y_0 < y_1 :
            y_points = np.linspace(y_0,y_1, num=20).tolist()
        if y_0 > y_1 :
            y_points = np.linspace(y_1,y_0, num=20).tolist()
        for y in y_points:
            line.append([x_0,y])
    return line
# Define points p_1, p_2, p_3, p_4 for obstacles
p_1, p_2, p_3, p_4 = O[0], O[1], O[2], O[3]
# Add radius to the first 4 points to create obstacle circles
for obstacle in range(len(O)) : #add to the first 4 points the radius
    Obstacles_for_RRT[obstacle].append(0.12)
# Create line obstacles between points p1-p2 and p3-p4 and add them to the obstacle list
line_between_1_2 = create_line_obstacle(p_1, p_2)
line_between_3_4 = create_line_obstacle(p_3, p_4)
for points in line_between_1_2 :
    Obstacles_for_RRT.append([points[0], points[1], 0.12])
for points in line_between_3_4 :
    Obstacles_for_RRT.append([points[0], points[1], 0.12])
# Convert obstacle list to the required format (x, y, radius)
final_O = []

for i in Obstacles_for_RRT :
    final_O.append((i[0],i[1],i[2]))

########################
#######################

# TO SHOW THE WALLS BETWEEN THE POINTS

# new_O = []
# for i in Obstacles_for_RRT :
#     new_O.append((i[0],i[1]))
# O = np.array(new_O)

########################
#######################

#start position 
Pc = [x_r[0],x_r[1]]

#RRT algorithm :
show_animation = True
class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])+0.005
            self.xmax = float(area[1])-0.005
            self.ymin = float(area[2])+0.005
            self.ymax = float(area[3])-0.005


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=None, #Not relevant to change it here, if you want to change it, change it at line "path = planner(Pc, p_g, final_O, B = [0, 2.4],expand_dist = *********"
                 path_resolution=None, #Not relevant to change it here, if you want to change change it in delta at line "path = planner(Pc, p_g, final_O, B = [0, 2.4],expand_dist = 12, delta=*********"
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=[0,2.4,0,2.4],
                 robot_radius=None,
                 ):

        print("path resolution is : "+str(path_resolution))
        print("expend dist is : "+str(expand_dis))

        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]+0.005
        self.max_rand = rand_area[1]-0.005
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation=True,num = None):
        """

        RRT (Rapidly-exploring Random Trees) path planning algorithm.
        
        Parameters:
        - animation (bool): If True, the planning process will be animated.
        - num (int): Number of iterations to run the RRT algorithm.

        Returns:
        - path (list of lists): The path generated by the RRT algorithm.
                                Each element of the list is a coordinate [x, y].
                                If the goal is not reachable, returns an empty list.
                                
        """
         # Initialize the node list with the start node
        self.node_list = [self.start]

        # Iterate the RRT algorithm for the given number of iterations
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            # Steer the nearest node towards the random node with a given expansion distance
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Check if the new_node is within the allowed play area and doesn't collide with obstacles
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)
            # Check if the final node is within the expansion distance to the goal 
            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                 # Check if the final node doesn't collide with obstacles
                if self.check_collision(
                        final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Steer from the 'from_node' towards the 'to_node' with a maximum extension length.
        The method creates a new node by extending the path from 'from_node' to 'to_node' with the given length.
        
        Parameters:
        - from_node (Node): The current node from which to steer.
        - to_node (Node): The target node towards which to steer.
        - extend_length (float): The maximum length to extend the path. Default is infinity.
        
        Returns:
        - new_node (Node): The newly created node after steering towards the 'to_node'.
        """

        # Initialize a new node with the same position as the 'from_node'
        new_node = self.Node(from_node.x, from_node.y)

        # Calculate the distance and angle between 'from_node' and 'to_node'
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        # Initialize the path lists of the new node with its initial position
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        # Limit the extension length to the given value or the actual distance if it is smaller
        if extend_length > d:
            extend_length = d

        # Calculate the number of expansions to reach the maximum extension length
        n_expand = math.floor(extend_length / self.path_resolution)

        # Extend the path from 'from_node' to 'to_node' by 'n_expand' steps
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        # Check if the new node is very close to the 'to_node', in which case set the new node to the 'to_node'
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        # Set the parent of the new node as the 'from_node'
        new_node.parent = from_node

        return new_node


    def generate_final_course(self, goal_ind):
        """
        Generate the final path from the goal node to the start node by backtracking through the parent nodes.
        
        Parameters:
        - goal_ind (int): Index of the goal node in the node list.
        
        Returns:
        - path (list of lists): The final path from the goal to the start node, represented as a list of coordinates [x, y].
        """

        # Initialize the path with the goal node's position
        path = [[self.end.x, self.end.y]]

        # Start from the goal node and backtrack through the parent nodes until reaching the start node
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent

        # Append the start node's position to complete the path
        path.append([node.x, node.y])

        return path


    def calc_dist_to_goal(self, x, y):
        """
        Calculate the Euclidean distance from the given (x, y) position to the goal (self.end.x, self.end.y).
        
        Parameters:
        - x (float): x-coordinate of the position to calculate the distance from.
        - y (float): y-coordinate of the position to calculate the distance from.
        
        Returns:
        - distance (float): The Euclidean distance from the (x, y) position to the goal.
        """

        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)


    def get_random_node(self):
        """
        Generate a random node within the specified bounds or sometimes at the goal itself with a certain probability.
        
        Returns:
        - rnd (Node): The randomly generated node.
        """

        # Randomly decide whether to choose a random node within bounds or sometimes the goal node
        if random.randint(0, 100) > self.goal_sample_rate:
            # Generate a random node within the specified bounds
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)
            )
        else:
            # Choose the goal node itself as the random node sometimes
            rnd = self.Node(self.end.x, self.end.y)

        return rnd


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """
        Get the index of the nearest node in the 'node_list' to the 'rnd_node'.

        Parameters:
        - node_list (list of Node): The list of nodes to find the nearest node in.
        - rnd_node (Node): The random node to find the nearest node to.

        Returns:
        - minind (int): The index of the nearest node in the 'node_list'.
        """

        # Calculate the squared distances from 'rnd_node' to each node in 'node_list'
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                for node in node_list]

        # Find the index of the node in 'node_list' with the minimum squared distance
        minind = dlist.index(min(dlist))

        return minind


    @staticmethod
    def check_if_outside_play_area(node, play_area):
        """
        Check if the given 'node' is outside or inside the specified 'play_area'.

        Parameters:
        - node (Node): The node to check for being inside or outside the 'play_area'.
        - play_area (AreaBounds or None): The bounding area to check the 'node' against.
                                        If 'play_area' is None, every position is considered valid.

        Returns:
        - (bool): True if 'node' is inside the 'play_area', False if it's outside.
        """

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin+0.084 or node.x > play_area.xmax-0.084 or \
        node.y < play_area.ymin+0.084 or node.y > play_area.ymax+0.084:
            return False  # outside - bad
        else:
            return True  # inside - ok


    @staticmethod
    def check_collision(node, obstacleList, robot_radius):
        """
        Check if the path of the 'node' collides with any of the obstacles in 'obstacleList'.

        Parameters:
        - node (Node): The node to check for path collision.
        - obstacleList (list of tuple): The list of obstacles in the form (ox, oy, size),
                                        where 'ox' and 'oy' are obstacle positions, and 'size' is its radius.
        - robot_radius (float): The radius of the robot.

        Returns:
        - (bool): True if the path of 'node' is collision-free, False if it collides with any obstacle.
        """

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            # Calculate the squared distances from each point in the node's path to the obstacle's center (ox, oy)
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            # Check if any point in the node's path is within the sum of the obstacle size and robot radius squared
            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision

        return True  # safe


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        Calculate the Euclidean distance and angle from 'from_node' to 'to_node'.

        Parameters:
        - from_node (Node): The starting node.
        - to_node (Node): The destination node.

        Returns:
        - d (float): The Euclidean distance between 'from_node' and 'to_node'.
        - theta (float): The angle (in radians) between the line connecting 'from_node' and 'to_node'
                        and the x-axis.
        """

        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


########################################
# get smoother path

def get_path_length(path):
    """
    Calculate the length of a path represented as a list of points.

    Parameters:
    - path (list): A list of points, where each point is represented as [x, y].

    Returns:
    - length (float): The total length of the path.
    """
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d

    return le

def get_target_point(path, targetL):
    """
    Find a point on a given path that corresponds to a specified length along the path.

    Parameters:
    - path (list): A list of points, where each point is represented as [x, y].
    - targetL (float): The target length along the path to find the corresponding point.

    Returns:
    - target_point (list): The point on the path that corresponds to the target length.
      The point is represented as [x, y], and the index of the point on the path is also returned.
    """
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break

    partRatio = (le - targetL) / lastPairLen

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

    return [x, y, ti]

def line_collision_check(first, second, obstacleList):
    """
    Check if a line segment between two points collides with any obstacles.

    Parameters:
    - first (list): The starting point of the line segment, represented as [x, y].
    - second (list): The ending point of the line segment, represented as [x, y].
    - obstacleList (list): A list of obstacles, where each obstacle is represented as (ox, oy, size).

    Returns:
    - collision (bool): True if the line segment collides with any obstacle, False otherwise.
    """
    # Line Equation

    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]

    try:
        a = y2 - y1
        b = -(x2 - x1)
        c = y2 * (x2 - x1) - x2 * (y2 - y1)
    except ZeroDivisionError:
        return False

    for (ox, oy, size) in obstacleList:
        d = abs(a * ox + b * oy + c) / (math.hypot(a, b))
        if d <= size+0.1:
            return False

    return True  # OK

def path_smoothing(path, max_iter, obstacle_list):
    """
    Smooth a given path by iteratively adjusting the path to avoid obstacles.

    Parameters:
    - path (list): A list of points representing the initial path, where each point is [x, y].
    - max_iter (int): The maximum number of iterations for the path smoothing process.
    - obstacle_list (list): A list of obstacles, where each obstacle is represented as (ox, oy, size).

    Returns:
    - smoothed_path (list): The smoothed path after avoiding obstacles, represented as a list of points.
    """
    le = get_path_length(path)

    for i in range(max_iter):
        # Sample two points
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])

        if first[2] <= 0 or second[2] <= 0:
            continue

        if (second[2] + 1) > len(path):
            continue

        if second[2] == first[2]:
            continue

        # collision check
        if not line_collision_check(first, second, obstacle_list):
            continue

        # Create New path
        newPath = []
        newPath.extend(path[:first[2] + 1])
        newPath.append([first[0], first[1]])
        newPath.append([second[0], second[1]])
        newPath.extend(path[second[2] + 1:])
        path = newPath
        le = get_path_length(path)

    return path

########################################
def planner(Pc, Pg, O, B=[0, 2.4],expand_dist = None, delta= None, show_graph=False):
    """
    Path planning function that uses the RRT algorithm to find a path from 'Pc' to 'Pg' while avoiding obstacles.

    Parameters:
    - Pc (list of float): The start position coordinates [x, y] of the robot.
    - Pg (list of float): The goal position coordinates [x, y] to reach.
    - O (list of tuples): The list of obstacles, where each tuple is (ox, oy, size).
                          'ox' and 'oy' are the obstacle positions, and 'size' is its radius.
    - B (list of float): The boundary of the map area in the form [xmin, xmax, ymin, ymax].
                         Default is [0, 2.4, 0, 2.4].
    - delta (float): The path resolution or step size for the RRT algorithm. Default is 0.02.
    - show_graph (bool): If True, the planning process will be animated to visualize the RRT. Default is False.

    Returns:
    - path (list of lists): The planned path from 'Pc' to 'Pg' represented as a list of coordinates [x, y].
                            The path will be reversed to start from the 'Pc' position.
                            If no valid path is found, it returns an empty list.
    """

    # Set the radius to take for collision checking
    radius_to_take = 0.07

    # Initialize the RRT algorithm with the given parameters
    rrt = RRT(start=Pc, goal=Pg, obstacle_list=O, rand_area=B,
              expand_dis=expand_dist, path_resolution=delta, play_area=[0.005, 2.395, 0.005, 2.395], robot_radius=radius_to_take)

    # Run the RRT algorithm to find the path
    path = rrt.planning(animation=show_graph)

    # Check if a valid path is found
    if path is None:
        print("Cannot find path")
        return []
    else:
        # Reverse the path to start from the 'Pc' position
        return path[::-1]


path = planner(Pc, p_g, final_O, B = [0, 2.4],expand_dist = 0.6, delta=0.05, show_graph=True) # the delta here is the path resolution
# print(path)
maxIter = 1000
smoothedPath = path_smoothing(path, maxIter, final_O)
# print(smoothedPath)

W = WheeledCar(x = x_r, goal = p_g, Obs = O) # Initiate simulation with x0, goak and Obs

# Get status of car - use these parameters in your planning
x_r, p_g, p_1, p_2, p_3, p_4 = W.field_status()
path = np.array(path)
smoothedPath = np.array(smoothedPath)

# # Run the robot along the planned path
W.run(smoothedPath) #  to run with RRT* (without using smoother path)
# W.run(path) # to run only with RRT