import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

def calculate_trajectory(initial_velocity, angle, initial_height):
    """
    Calculates the trajectory of a projectile.

    Args:
        initial_velocity (float): The initial velocity in m/s.
        angle (float): The launch angle in degrees.
        initial_height (float): The initial height in meters.

    Returns:
        tuple: A tuple containing time, x_coords, and y_coords arrays.
    """
    g = 9.81  # Acceleration due to gravity (m/s^2)
    angle_rad = np.radians(angle)

    # Initial velocity components
    v0_x = initial_velocity * np.cos(angle_rad)
    v0_y = initial_velocity * np.sin(angle_rad)

    # Time of flight calculation
    # Using the quadratic formula to find when y(t) = 0
    # y(t) = y0 + v0y*t - 0.5*g*t^2
    a = -0.5 * g
    b = v0_y
    c = initial_height
    
    # Check if the projectile is launched from the ground
    if initial_height <= 0:
        time_of_flight = (2 * v0_y) / g
    else:
        # Use quadratic formula for time to hit the ground (y=0)
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
             # This case shouldn't happen in a typical projectile problem
            time_of_flight = (2 * v0_y) / g 
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            time_of_flight = max(t1, t2)

    # Generate time points
    t = np.linspace(0, time_of_flight, num=500)

    # Calculate x and y coordinates
    x = v0_x * t
    y = initial_height + v0_y * t - 0.5 * g * t**2

    return t, x, y

def create_animation(params):
    """
    Creates a Matplotlib animation of the projectile motion.

    Args:
        params (dict): A dictionary with keys 'initial_velocity', 'angle', 'initial_height'.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    try:
        t, x, y = calculate_trajectory(
            params['initial_velocity'],
            params['angle'],
            params['initial_height']
        )
    except KeyError:
        # Return a blank figure if params are invalid
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid parameters for simulation.", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set plot limits
    ax.set_xlim(0, np.max(x) * 1.1)
    ax.set_ylim(0, np.max(y) * 1.1)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Motion Simulation")
    ax.grid(True)

    # The line object that will be updated in the animation
    line, = ax.plot([], [], 'o-', lw=2, color='blue', label='Trajectory')
    projectile, = ax.plot([], [], 'o', color='red', markersize=10)
    
    # Text for displaying stats
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        """Initialize the animation elements."""
        line.set_data([], [])
        projectile.set_data([], [])
        time_text.set_text('')
        return line, projectile, time_text

    def animate(i):
        """Perform animation step."""
        # Update the trajectory line up to the current frame
        line.set_data(x[:i], y[:i])
        
        # Update the position of the projectile (the red dot)
        projectile.set_data(x[i], y[i])
        
        # Update the time text
        time_text.set_text(time_template % t[i])
        
        return line, projectile, time_text

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t), init_func=init, blit=True, interval=20
    )

    return fig
