import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter

class ArrayVisualizer:
    """Visualizes arrays as raster images with consistent scale and animations.

    Attributes:
        colormap (str): Colormap to use for visualizing arrays.
        fixed_scale (tuple): Optional fixed scale (vmin, vmax) for normalization.
    """

    def __init__(self, colormap='viridis', fixed_scale=None):
        """Initializes the ArrayVisualizer with default colormap and fixed scale.

        Args:
            colormap (str): Colormap for the visualizations.
            fixed_scale (tuple, optional): Fixed scale for normalization as (vmin, vmax).
        """
        self.colormap = colormap
        self.fixed_scale = fixed_scale
        

    def plot_array(self, array, x_coords=None, y_coords=None, show_coords=True, blank_zeros=True):
        if blank_zeros:
            plot_array = np.ma.masked_where(array == 0, array)
            max_val = plot_array.max()
        else:
            plot_array = array
            max_val = np.max(array)

        norm = Normalize(vmin=self.fixed_scale[0], vmax=self.fixed_scale[1]) if self.fixed_scale else Normalize(vmin=plot_array.min(), vmax=max_val)

        fig, ax = plt.subplots()
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()] if x_coords is not None and y_coords is not None else None
        cax = ax.imshow(plot_array, cmap=self.colormap, norm=norm, extent=extent)

        if show_coords and x_coords is not None and y_coords is not None:
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            plt.xticks(rotation=45)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Find the position of the maximum value for labeling
        max_loc = np.unravel_index(np.argmax(plot_array), plot_array.shape)
        if extent is not None:
            max_x, max_y = x_coords[max_loc[1]], y_coords[max_loc[0]]
        else:
            max_x, max_y = max_loc[1], max_loc[0]
        ax.annotate(f'Max: {max_val}', xy=(max_x, max_y), color='black', weight='bold', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

        fig.colorbar(cax, ax=ax)
        plt.show()

        # Print the maximum value found in the array for cross-checking
        print(f'Maximum value in the array: {max_val}')


    def plot_multiple_arrays(self, arrays, x_coords=None, y_coords=None, show_coords=True):
        """Plots a list of 2D NumPy arrays as raster images.

        This method plots each array in the given list of arrays on separate figures. An option is provided to
        show or hide the XY coordinates on the axis.

        Args:
            arrays (list of np.ndarray): List of 2D arrays to visualize.
            x_coords (np.ndarray, optional): X coordinates for custom axis scaling.
            y_coords (np.ndarray, optional): Y coordinates for custom axis scaling.
            show_coords (bool): Whether to show coordinates on the axis.
        """
        if not self.fixed_scale:
            global_min = min(arr.min() for arr in arrays)
            global_max = max(arr.max() for arr in arrays)
            self.fixed_scale = (global_min, global_max)

        for idx, array in enumerate(arrays):
            print(f"Plotting Array {idx}")
            self.plot_array(array, x_coords, y_coords, show_coords)

    
    def animate_arrays(self, arrays, x_coords=None, y_coords=None, interval=200, repeat_delay=500, blank_zeros=True):
        """Creates an animation of a series of 2D NumPy arrays.

        Args:
            arrays (list of np.ndarray): List of 2D arrays to animate.
            x_coords (np.ndarray, optional): X coordinates for custom axis scaling.
            y_coords (np.ndarray, optional): Y coordinates for custom axis scaling.
            interval (int): Delay between frames in milliseconds.
            repeat_delay (int): Delay before repeating the animation in milliseconds.
            blank_zeros (bool): Whether to mask zero values in the arrays.

        Returns:
            FuncAnimation: The animation object.
        """
        fig, ax = plt.subplots()
        
        norm = Normalize(vmin=self.fixed_scale[0], vmax=self.fixed_scale[1]) if self.fixed_scale else Normalize(vmin=min(arr.min() for arr in arrays), vmax=max(arr.max() for arr in arrays))

        def update(frame):
            ax.clear()
            plot_array = np.ma.masked_where(arrays[frame] == 0, arrays[frame]) if blank_zeros else arrays[frame]
            ax.imshow(plot_array, cmap=self.colormap, norm=norm)
            ax.set_title(f'Frame {frame}')
            if x_coords is not None and y_coords is not None:
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                plt.xticks(rotation=45)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            return ax,

        ani = FuncAnimation(fig, update, frames=len(arrays), interval=interval, repeat_delay=repeat_delay, blit=False)
        
        return ani

    def save_visualization(self, array, file_name, x_coords=None, y_coords=None, show_coords=True, blank_zeros=True, dpi=300):
        """Saves a single array visualization as an image file.

        Args:
            array (np.ndarray): 2D array to visualize.
            file_name (str): The path to the output image file.
            x_coords (np.ndarray, optional): X coordinates for custom axis scaling.
            y_coords (np.ndarray, optional): Y coordinates for custom axis scaling.
            show_coords (bool): Whether to show coordinates on the axis.
            blank_zeros (bool): Whether to mask zero values in the array.
            dpi (int): Dots per inch setting for the output image.
        """
        self.plot_array(array, x_coords, y_coords, show_coords, blank_zeros)
        plt.savefig(file_name, dpi=dpi)
        plt.close()
    
    def save_animation(self, animation, file_name, fps=24, codec='libx264', extra_args=None):
        """Saves the animation as a video file.

        Args:
            animation (FuncAnimation): The animation object to save.
            file_name (str): The path to the output video file.
            fps (int): Frames per second of the output video.
            codec (str): The codec to use for writing the video.
            extra_args (list of str, optional): Additional arguments to pass to the video writer.
        """
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, codec=codec, extra_args=extra_args)
        animation.save(file_name, writer=writer)