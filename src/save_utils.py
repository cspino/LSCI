"""
Utils for LSCI video saving
"""
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import cv2
from tqdm import tqdm
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def get_final_shape(frame: np.ndarray, stack_max:int):
    # Get shape of video including colorbar
    DPI = 100
    figsize = (frame.shape[1]/DPI, frame.shape[0]/DPI)

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    im = ax.imshow(frame, cmap='plasma', vmin=0, vmax=stack_max)
    add_colorbar(im)
    ax.axis('off')

    fig.canvas.draw()
    frame_size = (int(fig.get_figwidth() * fig.get_dpi()),
                  int(fig.get_figheight() * fig.get_dpi()))
    plt.close(fig)
    print('frame size: ', frame_size)
    return frame_size, im, ax

def render_frame1(frame:np.ndarray, im, ax, stack_max)->np.ndarray:
    # DPI = 100
    # figsize = (frame.shape[1]/DPI, frame.shape[0]/DPI)
    # fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    im.set_array(frame)
    # ax.imshow(frame, cmap='plasma', vmin=0, vmax=stack_max)
    # # add_colorbar(im)
    # # im.set_array(frame)
    # # ax.axis('off')
    # plt.tight_layout()

    fig = ax.get_figure()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)
    return img

def _create_colorbar(height:int, width:int, cmap:str, frac:float=0.05, num_ticks=5)->np.ndarray:
    width = int(frac*width)
    gradient = np.linspace(255,0,height,dtype=np.uint8).reshape(-1,1)
    cmap = cv2.applyColorMap(gradient, getattr(cv2, f'COLORMAP_{cmap.upper()}'))
    cbar = cv2.resize(cmap, (width, height))
    return cbar
 
def create_colorbar(height, cmap:str, max_value=255):
    fig, ax = plt.subplots(figsize=(1,20))
    norm = mpl.colors.Normalize(vmin=0, vmax=max_value)
    print("cmap: ", cmap)
    print(mpl.cm.get_cmap(cmap))
    cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.get_cmap(cmap), norm=norm, orientation='vertical')
    fig.subplots_adjust(right=0.6, left=0.1, top=0.99, bottom=0.01)
    fig.canvas.draw()

    colorbar_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    colorbar_img = colorbar_img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    colorbar_img = cv2.resize(colorbar_img, (colorbar_img.shape[1], height))
    return cv2.cvtColor(colorbar_img, cv2.COLOR_RGB2BGR)

def save_flowmap(stack:np.ndarray, output_path:Path, fps:int, show:bool=False)->None:
    print("Saving relative flow map...")
    print('stack shape: ', stack.shape)

    colorbar = create_colorbar(stack.shape[1], "hot", stack.max())
    print(colorbar.shape)

    final_frame_size = (int(stack.shape[2]+colorbar.shape[2]), stack.shape[1]) # W, H
    # frame_size, im, ax = get_final_shape(stack[0], stack_max)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (final_frame_size[0], final_frame_size[1]))

    # Create the first frame and add the colorbar
    # DPI = 100
    # _, ax = plt.subplots(figsize=(stack[0].shape[1] / DPI, stack[0].shape[0] / DPI), dpi=DPI)
    # im = ax.imshow(stack[0], cmap='plasma', vmin=0, vmax=stack_max)
    # add_colorbar(im)  # Create the colorbar once
    # ax.axis('off')
    # colorbar = create_colorbar(stack.shape[1], stack.shape[0], 'hot', colorbar_frac)

    print('colorbar shape: ', colorbar.shape)

    stack = stack.astype(np.uint8)

    for i in tqdm(range(stack.shape[0]), desc="Saving video", disable=False):
        # time1 = time.time()
        # frame = render_frame(stack[i], im, ax, stack_max)
        time2 = time.time()
        # print('Render time: ', time2-time1)
        cmap_frame = cv2.applyColorMap(stack[i], cv2.COLORMAP_HOT)
        frame = np.hstack((cmap_frame, colorbar))
        timeee = time.time()
        # print('Render time: ', time2-timeee)

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Grayscale
        vidout = cv2.resize(frame, final_frame_size)
        time3 = time.time()
        # print('cv2 time: ', time3-timeee)
        video_writer.write(vidout)
        # print('Write time: ', time.time()-time3)

        if show:
            cv2.imshow('Frame', frame)

            # Wait for 1 ms and check if the user pressed the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the VideoWriter
    video_writer.release()
    cv2.destroyWindow('Frame')

def save_speckle(speckle:np.ndarray, file_path:Path, frame_rate:int, show:bool=False):
    n_frames, height, width = speckle.shape

    # Normalize between 0 and 255
    clip_value = 0.5*speckle.max() #TODO validate normalization
    speckle = np.clip(speckle, 0, clip_value)
    speckle = (speckle*255/speckle.max()).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter((file_path).as_posix(), fourcc, frame_rate, (width, height))

    for i in tqdm (range(n_frames), desc="Saving speckle video"):
        frame = speckle[i]

        # Convert grayscale to BGR by repeating channels
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Grayscale
        vidout=cv2.resize(frame,(width,height))
        video_writer.write(vidout)

        if show:
            cv2.imshow('Frame', frame)

            # Wait for 1 ms and check if the user pressed the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the VideoWriter
    cv2.destroyWindow('Frame')
    video_writer.release()