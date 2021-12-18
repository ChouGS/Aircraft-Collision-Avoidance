import random
import numpy as np
import cv2

from agents.aircraft import Aircraft

class Zone:
    zoom_ratio = 60
    def __init__(self, num_aircrafts, random_gen=True, aclist=None):
        # Air zone size
        self.h = 10
        self.w = 10

        # Whether aircrafts are generated randomly
        if random_gen:
            self.aclist = self.gen_aircrafts(num_aircrafts)
        else:
            assert aclist is not None and len(aclist) == num_aircrafts
            self.aclist = []
            for i in range(len(aclist)):
                src = aclist[i][0]
                dest = aclist[i][1]
                self.aclist.append(Aircraft(i, src, dest, num_aircrafts, self.w, self.h))

    def show(self):
        '''
            Plot the air zone and the planes in it.
        '''
        # 60 unit length in canvas = 1km

        # Backgrounds
        canvas = np.ones(((self.h + 2) * Zone.zoom_ratio, (self.w + 2) * Zone.zoom_ratio, 3), dtype=np.int32) * 100
        canvas[Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31, Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31] = 255

        # Planes
        ac_components = []
        for ac in self.aclist:
            ac_canvas = np.zeros(((self.h + 2) * Zone.zoom_ratio, (self.w + 2) * Zone.zoom_ratio, 3), dtype=np.int32)
            for i in range(1, self.h):
                cv2.line(ac_canvas,
                         (Zone.zoom_ratio, Zone.zoom_ratio * (1 + i)),
                         (Zone.zoom_ratio * (1 + self.h) + 1, Zone.zoom_ratio * (1 + i)),
                         [100, 100, 100],
                         2)
            for i in range(1, self.w):
                cv2.line(ac_canvas,
                         (Zone.zoom_ratio * (1 + i), Zone.zoom_ratio),
                         (Zone.zoom_ratio * (1 + i), Zone.zoom_ratio * (1 + self.w) + 1),
                         [100, 100, 100],
                         2)
            cv2.line(ac_canvas,
                     (Zone.zoom_ratio, Zone.zoom_ratio),
                     (Zone.zoom_ratio, Zone.zoom_ratio * (1 + self.w) + 1),
                     [30, 30, 30],
                     2)
            cv2.line(ac_canvas,
                     (Zone.zoom_ratio, Zone.zoom_ratio * 11 + 1),
                     (Zone.zoom_ratio * (1 + self.h) + 1, Zone.zoom_ratio * (1 + self.w) + 1),
                     [30, 30, 30],
                     2)
            cv2.line(ac_canvas,
                     (Zone.zoom_ratio * (1 + self.h) + 1, Zone.zoom_ratio * (1 + self.w) + 1),
                     (Zone.zoom_ratio * (1 + self.h) + 1, Zone.zoom_ratio),
                     [30, 30, 30],
                     2)
            cv2.line(ac_canvas,
                     (Zone.zoom_ratio * (1 + self.h) + 1, Zone.zoom_ratio),
                     (Zone.zoom_ratio, Zone.zoom_ratio),
                     [30, 30, 30],
                     2)
            if not ac.arrived:
                ac_canvas[int((ac.y + 0.5) * Zone.zoom_ratio):int((ac.y + 1.5) * Zone.zoom_ratio) + 1, 
                        int((ac.x + 0.5) * Zone.zoom_ratio):int((ac.x + 1.5) * Zone.zoom_ratio) + 1] = ac.danger_zone_color
            dest_x, dest_y = ac.destination
            ac_canvas[(dest_y + 1) * Zone.zoom_ratio - 10:(dest_y + 1) * Zone.zoom_ratio + 11, 
                      (dest_x + 1) * Zone.zoom_ratio - 10:(dest_x + 1) * Zone.zoom_ratio + 11] = ac.dest_color

            for history_pos in ac.path_history:
                x, y = history_pos
                cv2.circle(ac_canvas,
                           (int((x + 1) * Zone.zoom_ratio), int((y + 1) * Zone.zoom_ratio)),
                           int(0.04 * Zone.zoom_ratio),
                           ac.history_color,
                           int(0.04 * Zone.zoom_ratio),
                           cv2.LINE_AA)
            for future_pos in ac.path:
                x, y = future_pos
                cv2.circle(ac_canvas,
                           (int((x + 1) * Zone.zoom_ratio), int((y + 1) * Zone.zoom_ratio)),
                           int(0.04 * Zone.zoom_ratio),
                           ac.path_color,
                           int(0.04 * Zone.zoom_ratio),
                           cv2.LINE_AA)
            cv2.rectangle(ac_canvas,
                          (int((ac.x - 1.1) * Zone.zoom_ratio), int((ac.y - 1.1) * Zone.zoom_ratio)),
                          (int((ac.x + 3.1) * Zone.zoom_ratio), int((ac.y + 3.1) * Zone.zoom_ratio)),
                          ac.disp_color,
                          1,
                          cv2.LINE_AA)  
            cv2.circle(ac_canvas, 
                       (int((ac.x + 1) * Zone.zoom_ratio), int((ac.y + 1) * Zone.zoom_ratio)),
                       int(0.1 * Zone.zoom_ratio),
                       ac.disp_color,
                       int(0.1 * Zone.zoom_ratio),
                       cv2.LINE_AA)
            ac_components.append(ac_canvas)

        # Mix plane canvases into a single figure
        if len(self.aclist) == 2:
            ac_mask = cv2.addWeighted(ac_components[0], 0.5,
                                      ac_components[1], 0.5,
                                      100)
            ac_mask[ac_mask == 100] = 0
        if len(self.aclist) == 3:
            ac_mask = cv2.addWeighted(cv2.addWeighted(ac_components[0], 1/2,
                                                      ac_components[1], 1/2,
                                                      0), 2/3,
                                      ac_components[2], 1/3,
                                      100)
            ac_mask[ac_mask == 100] = 0
        
        ac_mask[ac_mask > 255] = 255
        white_mask = np.where(np.sum(ac_mask, 2) == 0)
        ac_mask[white_mask] = 255

        canvas[Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31, Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31] = \
            ac_mask[Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31, Zone.zoom_ratio - 30:-Zone.zoom_ratio + 31]

        return canvas

    def gen_aircrafts(self, num_aircrafts):
        '''
            Randomly generate airplanes.
        '''
        aircraft_list = []
        position_list = [(0, i) for i in range(1, self.h)] + \
                        [(i, 0) for i in range(1, self.w)] + \
                        [(self.w, i) for i in range(1, self.h)] + \
                        [(i, self.h) for i in range(1, self.w)]
        for id in range(num_aircrafts):
            # Generate begin and end points for each aircrafts
            while True:
                begin_pos = random.choice(position_list)
                end_pos = random.choice(position_list)

                # To make things nontrivial, assert begin and end position 
                # do not appear on the same side of the ir zone
                if begin_pos[0] == 0 and end_pos[0] == 0:
                    continue
                if begin_pos[0] == self.w and end_pos[0] == self.h:
                    continue
                if begin_pos[1] == 0 and end_pos[1] == 0:
                    continue
                if begin_pos[1] == self.h and end_pos[1] == self.h:
                    continue

                # Assert that no two aircrafts share the same begin position
                if len(aircraft_list) > 0:
                    valid = True
                    for aircraft in aircraft_list:
                        if aircraft.source == begin_pos:
                            valid = False
                            break
                    if not valid:
                        continue

                aircraft_list.append(Aircraft(id, begin_pos, end_pos, num_aircrafts, self.w, self.h))
                break
            
        return aircraft_list
