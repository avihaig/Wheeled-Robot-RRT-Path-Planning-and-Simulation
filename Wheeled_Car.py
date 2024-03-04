import numpy as np
import matplotlib.pyplot as plt
import time


class WheeledCar:
    w = 0.1
    v = 0.05
    L = 0.7
    r = 0.07
    r_reach = 0.07
    r_obs = 0.12
    w_pivot = 0.12
    noise = 1.0
    x_max = 2.4
    y_max = 2.4
    prev_collision = False
    collision_count = 0

    def __init__(self, x = None, goal = None, Obs = []):
        self.O = []
        if x is None or goal is None:
            self.init_race()
        else:
            self.x, self.goal, self.O = x, goal, Obs
        self.X = [np.copy(self.x)]
        self.path = None
        
    def init_race(self):
        
        self.goal = np.array([np.random.random()*(self.x_max-self.r) + self.r, np.random.random()*(self.y_max-self.r) + self.r])

        while 1:
            self.x = np.array([np.random.random()*(self.x_max-2*self.r) + self.r, np.random.random()*(self.y_max-2*self.r) + self.r, np.random.random()*2*np.pi])
            if np.linalg.norm(self.x[:2] - self.goal) > 1:
                break

        for _ in range(4):
            k = True
            while k:
                o = np.array([np.random.random()*(self.x_max-self.r) + self.r, np.random.random()*(self.y_max-self.r) + self.r])
                if np.linalg.norm(o - self.goal) > 3*self.r and np.linalg.norm(o - self.x[:2]) > 3*self.r:
                    if len(self.O) == 0:
                        break
                    for oo in self.O:
                        if np.linalg.norm(o - oo) > 3*self.r:
                            k = False
            self.O.append(o)

    def step_forward(self, a):
        dx = self.v * np.cos(self.x[2])
        dy = self.v * np.sin(self.x[2])
        dtheta = self.v / self.L * np.tan(a)

        self.x += np.array([dx, dy, dtheta])
        self.X.append(np.copy(self.x))
    
    def step_pivot(self, a): 
        self.x[2] += np.sign(a) * self.w_pivot
        self.x[2] = self.fix_angle(self.x[2])
        self.X.append(np.copy(self.x))

    def plot(self, show = False):
        x = np.copy(self.x[:2])
        theta = self.x[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        fig = plt.figure(0, figsize = (8, 8))
        plt.clf()
        ax = fig.add_subplot(1, 1, 1)

        gg = plt.Circle((self.goal[0], self.goal[1]), self.r, facecolor = 'cyan')
        ax.add_patch(gg)

        for i, o in enumerate(self.O):
            oo = plt.Circle((o[0], o[1]), self.r_obs, facecolor = 'gray')
            ax.add_patch(oo)
            ax.text(o[0]-self.r_obs/5, o[1]-self.r_obs/5, str(i+1), bbox=dict(facecolor='black', alpha=0.), fontdict=dict(size=16))
        #####################################################################################################################################

        # Obstacle_list = 
        # for o in Obstacle_list:
        #     oo = plt.Circle((o[0], o[1]), self.r_obs, facecolor = 'gray')
        #     ax.add_patch(oo)

        #####################################################################################################################################

        if self.path is not None:
            ax.plot(self.path[:,0], self.path[:,1], '.-k')

        Xx = np.array(self.X)
        ax.plot(Xx[:,0], Xx[:,1], ':k')

        p = np.array([[x[0]-self.w/2, x[1]-self.w/2], [x[0]+self.w/2, x[1]-self.w/2+0.02], [x[0]+self.w/2, x[1]+self.w/2-0.02], [x[0]-self.w/2, x[1]+self.w/2]])
        p = R.dot((p-x).T).T + x
        w1 = np.array([[x[0]-self.w/4, x[1]+self.w/2-0.003], [x[0]-self.w/4, x[1]+self.w/2+0.02], [x[0]+self.w/4, x[1]+self.w/2+0.02], [x[0]+self.w/4, x[1]+self.w/2-0.003]])
        w1 = R.dot((w1-x).T).T + x
        w2 = np.array([[x[0]-self.w/4, x[1]-self.w/2+0.003], [x[0]-self.w/4, x[1]-self.w/2-0.02], [x[0]+self.w/4, x[1]-self.w/2-0.02], [x[0]+self.w/4, x[1]-self.w/2+0.003]])
        w2 = R.dot((w2-x).T).T + x

        pp = plt.Polygon(p, facecolor='yellow', edgecolor = 'black')
        ax.add_patch(pp)
        ww1 = plt.Polygon(w1, facecolor='black', edgecolor = 'black')
        ax.add_patch(ww1)
        ww2 = plt.Polygon(w2, facecolor='black', edgecolor = 'black')
        ax.add_patch(ww2)

        ax.set_title(f'Position: ({np.round(self.x[0], 2)},{np.round(self.x[1], 2)}), Angle: {np.round(np.rad2deg(self.x[2]), 2)} deg') 
        if self.prev_collision:
            ax.text(0.05, 2.2, 'COLLISION!', fontdict=dict(size = 20, color = 'red'))    
            
        ax.set_xlim([0, self.x_max])
        ax.set_ylim([0, self.y_max])
        ax.set_aspect('equal')

        if show:
            plt.show()

    def fix_angle(self, a):
        a -= 2*np.pi if a > np.pi else 0
        a += 2*np.pi if a < -np.pi else 0
        return a

    def check_collisions(self):
        if self.x[0] > self.x_max-self.r/2 or self.x[0] < 0+self.r/2:
            return True
        
        if self.x[1] > self.y_max-self.r/2 or self.x[1] < 0+self.r/2:
            return True

        for o in self.O:
            if np.linalg.norm(o - self.x[:2]) < self.r_obs + self.r/2:
                return True
            
        return False
    
    def angle_distance(self, a, b):
        if np.sign(a)*np.sign(b) < 0 and np.abs(a) > np.pi/2 and np.abs(b) > np.pi/2:
            return (np.pi - np.abs(a)) + (np.pi - np.abs(b))
        else:
            return np.abs(a-b)
    
    def run(self, path):
        self.collision_count = 0
        self.path = path
        i = 1
        angle_noise = np.random.random()*2*0.2-0.2
        self.prev_collision = False
        self.start_time = time.time()
        while 1:
            p = path[i]
            alpha = self.fix_angle(np.arctan2(p[1]-self.x[1], p[0]-self.x[0]) + angle_noise)

            if self.angle_distance(self.x[2], alpha) > np.deg2rad(5):
                a = np.array([np.cos(self.x[2]), np.sin(self.x[2])])
                b = np.array([np.cos(alpha), np.sin(alpha)])
                g = np.sign(np.cross(a,b))
                self.step_pivot(g)
            else:
                self.step_forward(np.random.random()*2*self.noise - self.noise)
                if np.linalg.norm(self.x[:2] - p) < self.r_reach:
                    i += 1
                    angle_noise = np.random.random()*2*0.2-0.2
                    if np.linalg.norm(self.x[:2] - self.goal) < self.r:
                        self.run_time = time.time()-self.start_time
                        print(f'Reached goal in {np.round(self.run_time, 2)} seconds with {self.collision_count} collisions!')
                        plt.plot()
                        break
            
            col = self.check_collisions()
            if col and not self.prev_collision:
                self.collision_count += 1
                print('Collision!')
                self.prev_collision = True
            elif not col:
                self.prev_collision = False

            self.plot()
            plt.pause(0.00001)
        print('Finished path')
        plt.show()

        return self.run_time, self.collision_count

    def field_status(self):
        x_r = self.x
        p_g = self.goal
        p_1 = self.O[0]
        p_2 = self.O[1]
        p_3 = self.O[2]
        p_4 = self.O[3]
        
        return x_r, p_g, p_1, p_2, p_3, p_4