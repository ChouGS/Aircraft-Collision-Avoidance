from typing import Type
import numpy as np
import cv2
import os
import shutil
import copy

from agents.zone import Zone
from agents.aircraft import Aircraft
from results import Recorder

os.makedirs('results', exist_ok=True)
simulation_id = 0

def willCollide(z, j, k):
    for i in range(min(len(z.aclist[j].path), len(z.aclist[k].path))):
        if z.aclist[j].path[i][0] == z.aclist[k].path[i][0] and \
           z.aclist[j].path[i][1] == z.aclist[k].path[i][1]:
            return True
        if i < min(len(z.aclist[j].path), len(z.aclist[k].path)) - 1:
            if z.aclist[j].path[i+1][0] == z.aclist[k].path[i][0] and \
               z.aclist[j].path[i+1][1] == z.aclist[k].path[i][1] and \
               z.aclist[j].path[i][0] == z.aclist[k].path[i+1][0] and \
               z.aclist[j].path[i][1] == z.aclist[k].path[i+1][1]:  
                return True

    return False

# Results recorder
rc = Recorder('results/results.txt')

while simulation_id < 1000:
    # Keep these lines for testing specific cases
    # aclist = [[(1, 0), (10, 7)], [(0, 9), (10, 4)], [(1, 10), (10, 2)]]
    # zone = Zone(num_aircrafts=3, random_gen=False, aclist=aclist)
    zone = Zone(num_aircrafts=3, random_gen=True)

    # Guarantee that there will be many collisions (if planes are generated randomly) 
    if not willCollide(zone, 0, 1) or not willCollide(zone, 1, 2) or not willCollide(zone, 0, 2):
        continue

    print(f"Running case {simulation_id}...")

    # Save the same configuration for the finite-step tests
    zone_root = copy.deepcopy(zone)

    # Directories to write output figures and videos
    shutil.rmtree(f'results/{simulation_id}/Full_path/frames', ignore_errors=True)
    os.makedirs(f'results/{simulation_id}/Full_path', exist_ok=True)
    os.makedirs(f'results/{simulation_id}/Full_path/frames', exist_ok=True)
    vw = cv2.VideoWriter(f"results/{simulation_id}/demo_full.mp4",
                            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                            5,
                            (Zone.zoom_ratio * (zone.w + 2), Zone.zoom_ratio * (zone.h + 2)),
                            True)

    # Full-length test
    rc.add_key('Full')
    print('    Full length test running...')

    # Set clock
    tick = 0

    while True:
        # Plot current air zone
        canvas = zone.show()
        cv2.imwrite(f'results/{simulation_id}/Full_path/frames/{tick}.jpg', canvas)
        img = cv2.imread(f'results/{simulation_id}/Full_path/frames/{tick}.jpg')
        vw.write(img)

        # If all planes have arrived, exit
        done = True
        for i in range(len(zone.aclist)):
            if not zone.aclist[i].arrived:
                done = False
        if done:
            print(f"\tCase {simulation_id} successful.")
            break
        
        # Correspondence and collision avoidance occurs every 5 time steps
        if tick % 5 == 0:
            # SB
            for ac in zone.aclist:
                ac.broadcast()

            # IC
            for ac1 in zone.aclist:
                for ac2 in zone.aclist:
                    if ac1.id != ac2.id:
                        ac1.fetch(ac2)
            
            # PD
            for ac in zone.aclist:
                ac.checkMaxEta()
            
            # IC
            for ac1 in zone.aclist:
                for ac2 in zone.aclist:
                    if ac1.id != ac2.id:
                        ac1.fetch(ac2)

            # CD
            collision = False
            for ac in zone.aclist:
                coll, cid = ac.willCollide()
                collision = collision or coll

            # RP
            if collision:
                # Collision will occur. Each aircraft Modifies its path 
                # according to priority to avoid collision.
                all_okay = [False for _ in range(len(zone.aclist))]

                count = 0
                while count < 3:
                    for ac1 in zone.aclist:
                        all_okay[ac1.id] = ac1.modifyPath()
                        for ac2 in zone.aclist:
                            if ac1.id != ac2.id:
                                ac2.fetch(ac1)
                    if all(all_okay):
                        break

                    # Dead-end occurs, shuffle priority and redo
                    print("\tDead end occurs.")
                    for ac1 in zone.aclist:
                        for ac2 in zone.aclist:
                            if ac1.id != ac2.id and not all_okay[ac2.id]:
                                ac1.fetch(ac2, force_priority=True)
                    count += 1

                    # 3 redo fails: return with failure
                    if count == 3:
                        tick = 50000000
                        break
                if tick == 50000000:
                    rc[key_name].append(tick)
                    break

        # M
        for ac in zone.aclist:
            ac.move()

        tick += 1

    # Record results and do visualization
    rc['Full'].append(tick * Aircraft.speed)
    vw.release()
    cv2.destroyAllWindows()
    del zone

    # s-step test
    for s in range(10):
        steps = s + 1

        # Restore the same case as in full-length
        zone2 = copy.deepcopy(zone_root)
        for ac in zone2.aclist:
            ac.forecast_length = steps

        # Directories to write output figures and videos
        shutil.rmtree(f'results/{simulation_id}/{steps}_step/frames', ignore_errors=True)
        os.makedirs(f'results/{simulation_id}/{steps}_step', exist_ok=True)
        os.makedirs(f'results/{simulation_id}/{steps}_step/frames', exist_ok=True)
        vw = cv2.VideoWriter(f"results/{simulation_id}/demo_{steps}steps.mp4",
                                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                5,
                                (Zone.zoom_ratio * (zone2.w + 2), Zone.zoom_ratio * (zone2.h + 2)),
                                True)

        # s-step forecast
        key_name = f'{steps}_step'
        rc.add_key(key_name)
        print(f'    {steps} step test running...')

        # Set clock
        tick = 0

        while True:
            # Plot current air zone
            canvas = zone2.show()
            cv2.imwrite(f'results/{simulation_id}/{steps}_step/frames/{tick}.jpg', canvas)
            img = cv2.imread(f'results/{simulation_id}/{steps}_step/frames/{tick}.jpg')
            vw.write(img)

            # If all planes have arrived, exit
            done = True
            for i in range(len(zone2.aclist)):
                if not zone2.aclist[i].arrived:
                    done = False
            if done:
                print(f"\tCase {simulation_id} successful.")
                break
            
            # Correspondence and collision avoidance occurs every 5 time steps
            if tick % 5 == 0:
                # SB
                for ac in zone2.aclist:
                    ac.broadcast()

                # IC
                for ac1 in zone2.aclist:
                    for ac2 in zone2.aclist:
                        if ac1.id != ac2.id:
                            ac1.fetch(ac2)
                
                # PD
                for ac in zone2.aclist:
                    ac.checkMaxEta()
                
                # IC
                for ac1 in zone2.aclist:
                    for ac2 in zone2.aclist:
                        if ac1.id != ac2.id:
                            ac1.fetch(ac2)

                # CD
                collision = False
                for ac in zone2.aclist:
                    coll, cid = ac.willCollide()
                    collision = collision or coll

                # RP
                if collision:
                    # Collision will occur. Each aircraft Modifies its path 
                    # according to priority to avoid collision.
                    all_okay = [False for _ in range(len(zone2.aclist))]

                    count = 0
                    while count < 3:
                        for ac1 in zone2.aclist:
                            all_okay[ac1.id] = ac1.modifyPath()
                            for ac2 in zone2.aclist:
                                if ac1.id != ac2.id:
                                    ac2.fetch(ac1)
                        if all(all_okay):
                            break

                        # Dead-end occurs, shuffle priority and redo
                        print("\tDead end occurs.")
                        for ac1 in zone2.aclist:
                            for ac2 in zone2.aclist:
                                if ac1.id != ac2.id and not all_okay[ac2.id]:
                                    ac1.fetch(ac2, force_priority=True)
                        count += 1

                        # 3 redo fails: return with failure
                        if count == 3:
                            tick = 50000000
                            break
                    if tick == 50000000:
                        rc[key_name].append(tick)
                        break

            # M
            for ac in zone2.aclist:
                ac.move()

            tick += 1

        # Record results and do visualization
        rc[key_name].append(tick * Aircraft.speed)
        vw.release()
        cv2.destroyAllWindows()
        del zone2
    
    simulation_id += 1

# Output final results
rc.summarize()
