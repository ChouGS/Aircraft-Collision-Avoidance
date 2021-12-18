import numpy as np
import copy

class Aircraft:
    speed = 0.2
    def __init__(self, id, src, dest, num_acs, zone_w, zone_h):
        # ID of aircraft
        self.id = id

        # Source and destination
        self.source = src
        self.destination = dest

        # Current position
        self.x = src[0]
        self.y = src[1]

        # Current orientation
        self.orientation = self.getOrientation()

        # Whether has arrived
        self.arrived = False

        # Size of air zone
        self.zone_h = zone_h
        self.zone_w = zone_w

        # Number of aircraft in the air zone
        self.num_acs = len(self.recv_msg)

        # Estimated time of arrival
        self.eta = 0

        # History and future paths
        self.path_history = []
        self.path = self.autoGenPath(src, dest)

        # Priority list
        self.recognized_priority = None

        # Message to be broadcast and received
        self.bc_msg = None
        self.recv_msg = [None for _ in range(num_acs)]

        # Colors for plotting
        self.danger_zone_color, self.history_color, self.path_color, \
            self.dest_color, self.disp_color = self.genColor(id)

        # How many steps ahead should be broadcast, -1 means full-length
        self.forecast_length = -1
        
    def broadcast(self):
        '''
            Broadcast self status to others.
        '''
        self.bc_msg = {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'orientation': self.orientation,
            'eta': self.eta,
            'path': self.path if self.forecast_length == -1 else \
                    self.path[:int(round(self.forecast_length / self.speed))],
            'arrived': self.arrived,
            'dest': self.destination,
            'recognized_priority': self.recognized_priority
        }
    
    def checkMaxEta(self):
        '''
            Compare among ETAs to generate priority list.
        '''
        eta_list = []
        id_list = []
        for i in range(len(self.recv_msg)):
            if i == self.id:
                eta_list.append(self.eta)
                id_list.append(self.id)
                continue
            if self.recv_msg[i] is not None:
                eta_list.append(self.recv_msg[i]['eta'])
                id_list.append(i)

        self.recognized_priority = np.array(id_list)[np.argsort(-np.array(eta_list))].tolist()
        self.broadcast()

    def getOrientation(self):
        '''
            Get orientation vector according to the current and past locations.
        '''
        if len(self.path) < 2:
            return self.orientation
        dx = int(round((self.path[0][0] - self.x) / Aircraft.speed))
        dy = int(round((self.path[0][1] - self.y) / Aircraft.speed))
        return (dx, dy)

    def fetch(self, aircraft, force_priority=False):
        '''
            Fetch message broadcast by other planes.
            If force_priority is True, hard copy the priority.
        '''
        if int(max(abs(self.x - aircraft.x), abs(self.y - aircraft.y))) <= 2:
            self.recv_msg[aircraft.id] = aircraft.bc_msg
            if force_priority:
                self.recognized_priority = copy.deepcopy(aircraft.recognized_priority)
                for i in range(len(self.recv_msg)):
                    if self.recv_msg[i] is None and i != self.id and i in self.recognized_priority:
                        self.recognized_priority.remove(i)
        else:
            self.recv_msg[aircraft.id] = None

    def willCollide(self):
        '''
            Detect whether collision(s) will happen in the future.
        '''
        collide_id = []
        for msg in self.recv_msg:
            if msg is not None:
                for i in range(min(len(self.path), len(msg['path']))):
                    if self.path[i][0] == msg['path'][i][0] and \
                       self.path[i][1] == msg['path'][i][1]:
                        collide_id.append(msg['id'])
                        break
                    if i < min(len(self.path), len(msg['path'])) - 1:
                        if self.path[i+1][0] == msg['path'][i][0] and \
                           self.path[i+1][1] == msg['path'][i][1] and \
                           self.path[i][0] == msg['path'][i+1][0] and \
                           self.path[i][1] == msg['path'][i+1][1]:  
                            collide_id.append(msg['id'])
                            break

        return (False, collide_id) if len(collide_id) == 0 else (True, collide_id)

    def modifyPath(self):
        '''
            This airplane is going to collide with the airplane with id {id}! 
            Suggest a new path to avoid collision!
        '''

        def getorienid(orien):
            if orien[0] == 0 and orien[1] == 1:
                return 0
            if orien[0] == 0 and orien[1] == -1:
                return 1
            if orien[0] == 1 and orien[1] == 0:
                return 2
            if orien[0] == -1 and orien[1] == 0:
                return 3

        def getPreferenceList(cur_x, cur_y, dest_x, dest_y):
            dx = abs(cur_x - dest_x)
            dy = abs(cur_y - dest_y)

            if cur_x < dest_x and cur_y < dest_y:
                if dx > dy:
                    outp = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                else:
                    outp = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            if cur_x > dest_x and cur_y < dest_y:
                if dx > dy:
                    outp = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                else:
                    outp = [(0, 1), (-1, 0), (0, -1), (1, 0)]

            if cur_x < dest_x and cur_y > dest_y:
                if dx > dy:
                    outp = [(1, 0), (0, -1), (-1, 0), (0, 1)]
                else:
                    outp = [(0, -1), (1, 0), (0, -1), (1, 0)]
            
            if cur_x > dest_x and cur_y > dest_y:
                if dx > dy:
                    outp = [(-1, 0), (0, -1), (1, 0), (0, 1)]
                else:
                    outp = [(0, -1), (-1, 0), (0, 1), (1, 0)]

            if cur_x == dest_x:
                if cur_y > dest_y:
                    outp = [(0, -1), (-1, 0), (1, 0), (0, 1)]
                else:
                    outp = [(0, 1), (-1, 0), (1, 0), (0, -1)]

            if cur_y == dest_y:
                if cur_x > dest_x:
                    outp = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                else:
                    outp = [(1, 0), (0, -1), (0, 1), (-1, 0)]

            return outp

        if self.recognized_priority[0] == self.id:
            return True

        # Self status
        x_a = int(self.x)
        y_a = int(self.y)
        orientation_a = self.orientation
        dest_ax, dest_ay = self.destination

        eta = self.eta

        # state: (x_a, y_a, o_a, time_used, state_id, last_state_id)
        sid = 1
        ptr = 0
        BFS_queue = [(x_a, y_a, orientation_a, 0, 0, -1)]
        used_state = np.zeros((self.zone_h + 1, self.zone_w + 1, 4))

        dead_end = False

        # BFS cycle
        while True:
            if ptr >= len(BFS_queue):
                dead_end = True
                break

            # Fetch the first state to be searched
            state = BFS_queue[ptr]

            # Test if both aircrafts has reached their destination
            if state[0] == dest_ax and state[1] == dest_ay:
                break
            
            # Refresh pointer
            ptr += 1

            # Neither aircraft has arrived:
            # Try different move combinations
            ma_iter = getPreferenceList(state[0], state[1], dest_ax, dest_ay)
            for ma in ma_iter:
                # Aircrafts are not allowed to make U-turns
                if ma[0] + state[2][0] == 0 and ma[1] + state[2][1] == 0:
                    continue
                new_state = (state[0] + ma[0], 
                             state[1] + ma[1],
                             ma,
                             state[3] + 1,
                             sid,
                             state[4])
                
                # Bound check: 
                if new_state[0] < 0 or new_state[0] > self.zone_w or \
                   new_state[1] < 0 or new_state[1] > self.zone_h: 
                    continue
                
                # Used states are not considered
                orien_id = getorienid(new_state[2])
                if used_state[new_state[0], new_state[1], orien_id] == 1:
                    continue

                # Safety requirements: no collision allowed
                safe = True
                num_constraints = self.recognized_priority.index(self.id)
                for c in range(num_constraints):
                    cid = self.recognized_priority[c]
                    x_b = int(self.recv_msg[cid]['x'])
                    y_b = int(self.recv_msg[cid]['y'])
                    constraint_path = self.recv_msg[cid]['path']
                    if len(constraint_path) >= 5 * new_state[3]:
                        if new_state[0] == constraint_path[5 * new_state[3] - 1][0] and \
                           new_state[1] == constraint_path[5 * new_state[3] - 1][1]:
                            safe = False
                            break
                    if new_state[0] == x_b and new_state[1] == y_b and \
                       constraint_path[4][0] == state[0] and constraint_path[4][1] == state[1]:
                        safe = False
                        break
                    if len(constraint_path) >= 5 * new_state[3] and len(constraint_path) >= 10:
                        if new_state[0] == constraint_path[5 * new_state[3] - 6][0] and \
                           new_state[1] == constraint_path[5 * new_state[3] - 6][1] and \
                           state[0] == constraint_path[5 * new_state[3] - 1][0] and \
                           state[1] == constraint_path[5 * new_state[3] - 1][1]:
                            safe = False
                            break
                if not safe:
                    continue

                # Cost pruning: if time consumed is larger than current eta + 4, do not consider it as a good solution
                fastest_eta = abs(new_state[0] - dest_ax) + abs(new_state[1] - dest_ay)
                if new_state[3] + fastest_eta > eta + self.num_acs * 2 - 2:
                    continue                   

                # Considerable states are stored for further searching
                BFS_queue.append(new_state)
                sid += 1

                # Searched state are recorded
                used_state[new_state[0], new_state[1], orien_id] = 1

        # If dead_end occurs, do priority shuffle and return with failure
        if dead_end:
            sid = self.recognized_priority.index(self.id)
            self.recognized_priority = [self.id] + self.recognized_priority[:sid] + self.recognized_priority[sid + 1:]
            self.broadcast()
            return False

        # Track back to restore the full path
        suggested_path = []
        while ptr != -1:
            suggested_path = [BFS_queue[ptr]] + suggested_path
            ptr = BFS_queue[ptr][-1]
        
        # Build the path information
        suggested_a = []
        for i in range(len(suggested_path)):
            # First step: directly append the point 
            if i == 0:
                suggested_a.append((suggested_path[i][0], suggested_path[i][1]))
            # Not the first step: append interpolated points
            elif not (suggested_a[-1][0] == dest_ax and suggested_a[-1][1] == dest_ay):
                # Moving horizontally
                if suggested_a[-1][0] != suggested_path[i][0]:
                    direction = -1 if suggested_a[-1][0] > suggested_path[i][0] else 1
                    suggested_a = suggested_a + [(round(j * Aircraft.speed, 2), suggested_path[i][1]) \
                        for j in range(int(round((suggested_a[-1][0] + direction * Aircraft.speed) / Aircraft.speed)), 
                                       int(round((suggested_path[i][0] + direction * Aircraft.speed) / Aircraft.speed)),
                                       direction)]
                # Moving vertically
                else:
                    assert suggested_a[-1][1] != suggested_path[i][1]
                    direction = -1 if suggested_a[-1][1] > suggested_path[i][1] else 1
                    suggested_a = suggested_a + [(suggested_path[i][0], round(j * Aircraft.speed, 2)) \
                        for j in range(int(round((suggested_a[-1][1] + direction * Aircraft.speed) / Aircraft.speed)), 
                                       int(round((suggested_path[i][1] + direction * Aircraft.speed) / Aircraft.speed)),
                                       direction)]

        self.path = self.autoGenPath(self.source,
                                     self.destination,
                                     suggested_a[1:])
        self.broadcast()

        return True

    def autoGenPath(self, begin, end, default_path=[]):
        '''
            begin, end: (x, y) tuple, default begin and end position for the aircraft
            path: list of (x, y) tuple, a section of path the aircraft MUST take to avoid collision
            -----------------------------------------------
            Generate the shortest path from {begin} to {end}. The very beginning of this path must 
            be the beginning point of {default_path} if it is provided.
        '''

        # Changes the beginning point to the last position in {default_path} if it is provided
        if len(default_path) != 0:
            begin = default_path[-1]

        # Calculate the distance the aircraft has to travel in both x and y direction.
        # The aircraft by default first travels along the direction with the larger delta.
        delta_x = abs(begin[0] - end[0])
        delta_y = abs(begin[1] - end[1])

        if delta_x == 0 and delta_y == 0:
            self.eta = len(default_path)
            return default_path

        direction_x = -1 if begin[0] > end[0] else 1 if begin[0] < end[0] else 0
        direction_y = -1 if begin[1] > end[1] else 1 if begin[1] < end[1] else 0
        
        if delta_x > delta_y:
            path1 = [(round(i * Aircraft.speed, 2), begin[1]) \
                      for i in range(int(round((begin[0] + direction_x * Aircraft.speed) / Aircraft.speed)), 
                                     int(round((end[0] + direction_x * Aircraft.speed) / Aircraft.speed)), 
                                     direction_x)]
            path2 = []
            if delta_y > 0:
                path2 = [(end[0], round(i * Aircraft.speed, 2)) \
                          for i in range(int(round((begin[1] + direction_y * Aircraft.speed) / Aircraft.speed)), 
                                         int(round((end[1] + direction_y * Aircraft.speed) / Aircraft.speed)),
                                         direction_y)]
            path = default_path + path1 + path2

        else:
            path1 = [(begin[0], round(i * Aircraft.speed, 2)) \
                      for i in range(int(round((begin[1] + direction_y * Aircraft.speed) / Aircraft.speed)),
                                     int(round((end[1] + direction_y * Aircraft.speed) / Aircraft.speed)), 
                                     direction_y)]
            path2 = []
            if delta_x > 0:
                path2 = [(round(i * Aircraft.speed, 2), end[1]) \
                        for i in range(int(round((begin[0] + direction_x * Aircraft.speed) / Aircraft.speed)), 
                                       int(round((end[0] + direction_x * Aircraft.speed) / Aircraft.speed)), 
                                       direction_x)]
            path = default_path + path1 + path2
        
        self.eta = len(path)
        return path

    def genColor(self, id):
        # Generate display color for each aircraft
        assert id in [0, 1, 2]
        if id == 0:
            dzone_color = [255, 150, 150]
            hist_color = [255, 50, 50]
            path_color = [255, 50, 50]
            dest_color = [255, 100, 100]
            disp_color = [255, 0, 0]
        if id == 1:
            dzone_color = [150, 150, 255]
            hist_color = [50, 50, 255]
            path_color = [50, 50, 255]
            dest_color = [100, 100, 255]
            disp_color = [0, 0, 255]
        if id == 2:
            dzone_color = [150, 255, 150]
            hist_color = [50, 255, 50]
            path_color = [50, 255, 50]
            dest_color = [100, 255, 100]
            disp_color = [0, 255, 0]
        return dzone_color, hist_color, path_color, dest_color, disp_color

    def move(self):
        # Let the aircraft move for one timestep. If it reaches its destination, change its state {arrival}.
        if self.arrived:
            return
        assert len(self.path) > 0
        self.path_history.append((self.x, self.y))
        self.orientation = self.getOrientation()
        self.x, self.y = self.path[0]
        self.path = self.path[1:]
        self.eta = len(self.path)
        if self.x == self.destination[0] and self.y == self.destination[1]:
            assert self.eta == 0
            self.arrived = True
