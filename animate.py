import os
import sys
# pyplot3d lives in third_party/ (git submodule)
_THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

from matplotlib import animation

from pyplot3d.uav import Uav
from pyplot3d.utils import ypr_to_R
import pandas as pd



def animate_quadcopter_history(times, x, R,arm_length=0.3,tilt_angle=0.0):
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    horiz_lim = (-2,2)
    verti_lim = (0,6)
    uav_plot = Uav(ax, arm_length,tilt_angle=tilt_angle) 
    def update_plot(i):
        
        # ax.cla()
            
        uav_plot.draw_at(x[:, i], R[:, :, i])
        
        # These limits must be set manually since we use
        # a different axis frame configuration than the
        # one matplotlib uses.
        
        ax.set_xlim(horiz_lim)
        ax.set_ylim(horiz_lim)
        ax.set_zlim(verti_lim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        # ax.invert_zaxis()
        ax.set_title("Quadcopter Animation (Time: {0:.3f} s)".format(times[i]))
    
    # animate @ 1/desired_interval hz, with data from every step_size * dt
    animate_interval = 50
    step_size = 25
    steps = len(times)
    ind = [i * step_size for i in range(steps // step_size)]
    ani = animation.FuncAnimation(fig, update_plot,frames=ind,interval=animate_interval);
    # Define the filename for the GIF
    output_gif_filename = "quad_sim_animation.gif"
    # Save the animation as a GIF
    ani.save(output_gif_filename, writer='pillow')
    # Display the animation
    plt.show()

def animate_forward_flight_history(times, x, R, ref_point, ref_dir, speeds=None, arm_length=0.4,
                                   window=(4.0, 3.0), trail_time=1.0,
                                   gif_filename="forward_flight_animation.gif", reference=None):
    """Forward-flight animation in a frame that moves with the vehicle: the view stays centred on it, so the
    axes and the reference line scroll past while the vehicle shows its attitude.

    x: 3xN positions, R: 3x3xN attitudes, ref_point/ref_dir: the straight reference line (point, direction),
    speeds: optional N airspeeds for the title, window: (horizontal, vertical) half-widths of the view [m],
    trail_time: [s] of past path drawn as a thin red line (clipped to the view).
    reference: optional callable t -> (point, direction) for a reference line that changes over time; used in
    place of ref_point/ref_dir when given.
    The quadcopter model is a placeholder airframe for now."""
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    uav_plot = Uav(ax, arm_length)
    ref_point = np.asarray(ref_point, dtype=float)
    ref_dir = np.asarray(ref_dir, dtype=float) / np.linalg.norm(ref_dir)
    half_h, half_v = window
    half = np.array([half_h, half_h, half_v])
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    trail_steps = max(1, int(round(trail_time / dt)))

    def update_plot(i):
        centre = x[:, i]
        uav_plot.draw_at(centre, R[:, :, i])  # clears the axis, so everything else is drawn after it

        # reference line across the view: the part of the line next to the vehicle
        point, direction = (ref_point, ref_dir) if reference is None else reference(times[i])
        point = np.asarray(point, dtype=float)
        direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        s = np.dot(centre - point, direction)
        line = point[:, None] + direction[:, None] * np.array([s - half_h, s + half_h])
        ax.plot(line[0], line[1], line[2], '--', color='k', lw=1)

        # trail: recent path, only the part inside the view (3D axes don't clip)
        trail = x[:, max(0, i - trail_steps):i + 1]
        trail = trail[:, np.all(np.abs(trail - centre[:, None]) <= half[:, None], axis=0)]
        ax.plot(trail[0], trail[1], trail[2], color='r', lw=1)

        # view centred on the vehicle: the axes move, the vehicle stays in the middle
        ax.set_xlim(centre[0] - half_h, centre[0] + half_h)
        ax.set_ylim(centre[1] - half_h, centre[1] + half_h)
        ax.set_zlim(centre[2] - half_v, centre[2] + half_v)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([half_h, half_h, half_v])  # equal scale on all axes
        title = "Forward flight (time: {0:.3f} s".format(times[i])
        if speeds is not None:
            title += ", airspeed: {0:.1f} m/s".format(speeds[i])
        ax.set_title(title + ")")

    # same rate as animate_quadcopter_history: every step_size-th sample, one frame per animate_interval ms
    animate_interval = 50
    step_size = 25
    steps = len(times)
    ind = [i * step_size for i in range(steps // step_size)]
    ani = animation.FuncAnimation(fig, update_plot, frames=ind, interval=animate_interval)
    ani.save(gif_filename, writer='pillow')
    plt.show()

# # Create some fake simulation data
# steps = 500
# t_end = 1
# times = np.linspace(0, t_end, steps)
# x = np.zeros((3, steps))
# x[0, :] = np.arange(0, t_end, t_end / steps)
# x[1, :] = np.arange(0, t_end, t_end / steps) * 2

# R = np.zeros((3, 3, steps))
# for i in range(steps):
#     ypr = np.array([0, 0.5*i, 0.0])
#     R[:, :, i] = ypr_to_R(ypr, degrees=True)

# animate_quadcopter_history(times, x, R)