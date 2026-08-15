import time
import gym, gym_unrealcv
from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

env = gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0')
env = time_dilation.TimeDilationWrapper(env, 10)
env.unwrapped.agents_category = ['drone']
env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
env.seed(0)
env.reset()

unrealcv = env.unwrapped.unrealcv
drones = env.unwrapped.player_list
print('drones:', drones)

# reset() with this seed lands the drones at DowntownWest.json safe_start[15]
# ([-12060.906, -1723.04, -15.31]) - a negative-z interior/underground point, hence the
# wall collisions on the first try. safe_start[0] shares its z (160.685) with the JSON's
# top-level "height" field, i.e. it's the map's default outdoor spawn - teleport there.
#
# The previous attempt used character.py:set_cam to attach an offset camera to the flying
# drone, but its "loc" param behaves unpredictably (measured via get_camera_config(): a
# guess of [-1000,0,400] produced a near-zero world offset, not a simple scale factor -
# likely a spring-arm collision pulling the camera back near obstructions, or a
# rotation-dependent transform). That mechanism isn't debuggable without the UE blueprint
# source, so instead: leave drones[1] parked as a fixed external observer with its own
# *default* onboard camera (offset [0,0,0] - the one already proven to work) pointed at
# where drones[0] will fly. No attach/offset math involved.
open_air = [-10303.518, -2358.863, 160.685]
unrealcv.set_obj_location(drones[0], open_air)
unrealcv.set_obj_rotation(drones[0], [0, 0, 0])

# Empirically confirmed: at rotation [0,0,0], action v_x > 0 moves the drone toward
# world -x. Park the observer further +x ("behind", from where the flight departs),
# offset in y to avoid sitting exactly in the flight line, and higher up; same yaw=0
# rotation faces it the same direction the flying drone travels, tilted down to keep
# the flight path in frame.
observer_pos = [open_air[0] + 2500, open_air[1] + 400, open_air[2] + 900]
unrealcv.set_obj_location(drones[1], observer_pos)
unrealcv.set_obj_rotation(drones[1], [0, 0, -20])  # [roll, yaw, pitch]; pitch<0 tilts down
time.sleep(0.5)

unrealcv.set_viewport(drones[1])
print('flying drone start pose:', unrealcv.get_obj_location(drones[0]))
print('observer drone pose:', unrealcv.get_obj_location(drones[1]))

# The observer drifted into a wall last run despite getting a [0,0,0,0] "hover" action
# every step - it has no internal_nav/hover stabilization (DowntownWest.json), so a zero
# velocity command may not actually hold position against physics between the ~100ms-
# apart step calls, and/or the viewport itself gets reassigned by something in the
# per-step image capture. Re-pin position, rotation, and viewport every step to rule
# out both at once.
#
# action = [v_x, v_y, v_z, v_yaw] for drones[0], range [-1, 1] (character.py:set_move_bp).
# x-forward, y-right, z-up, yaw clockwise.
n_agents = len(drones)
def fly(steps, action_for_drone0):
    for _ in range(steps):
        actions = [[0, 0, 0, 0]] * n_agents
        actions[0] = action_for_drone0
        env.step(actions)
        unrealcv.set_obj_location(drones[1], observer_pos)
        unrealcv.set_obj_rotation(drones[1], [0, 0, -20])
        unrealcv.set_viewport(drones[1])

fly(15, [0.4, 0, 0.3, 0])   # forward + gentle climb, stays roughly level with the observer
fly(25, [0.3, 0, 0, 0.4])   # forward while turning
fly(15, [0, 0, 0, 0])       # hover so the final frame is easy to look at

print('flying drone end pose:', unrealcv.get_obj_location(drones[0]))
time.sleep(5)  # keep the window open long enough to actually look at it
env.close()
