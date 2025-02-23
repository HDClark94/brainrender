# Example code to plot trajectories in BrainRender from stereotaxic coordinates
import os
import math
import numpy as np
import pandas as pd
from brainrender import Scene
from brainrender.actors import Cylinder
from brainrender import VideoMaker
from brainrender import Animation
from brainrender import settings
from read_mat import read_probe_mat, read_borders_table
import spikeinterface.full as si


class Cylinder2(Cylinder):

    def __init__(self, pos_from, pos_to, root, color='powderblue', alpha=1, radius=350):
        from vedo import shapes
        from brainrender.actor import Actor
        mesh = shapes.Cylinder(pos=[pos_from, pos_to], c=color, r=radius, alpha=alpha)
        Actor.__init__(self, mesh, name="Cylinder", br_class="Cylinder")


# Function to convert stereotaxic coordinates to ABA CCF
# SC is an array with stereotaxic coordinates to be transformed
# Returns an array containing corresponding CCF coordinates in μm
# Conversion is from this post, which explains the opposite transformation: https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858/3
# Warning: this is very approximate
# Warning: the X, Y, Z schematic at the top of the linked post is incorrect, scroll down for correct one.
def StereoToCCF(SC = np.array([1,1,1]), angle = -0.0873):
    # Stretch
    stretch = SC/np.array([1,0.9434,1])
    # Rotate
    rotate = np.array([(stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle)),
                       (stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle)),
                       stretch[2]])
    #Translate
    trans = rotate + np.array([5400, 440, 5700])
    return(trans)
 
def CCFToStereo(CCF = np.array([1,1,1]), angle = 0.0873):
    #Translate
    trans = CCF - np.array([5400, 440, 5700])
    # Rotate
    rotate = np.array([(trans[0] * math.cos(angle) - trans[1] * math.sin(angle)),
                       (trans[0] * math.sin(angle) + trans[1] * math.cos(angle)),
                       trans[2]])
    # Stretch
    stretch = rotate*np.array([1,0.9434,1])
    return(stretch)

def adjust_border_table_for_apparent_longest_probe(probe_locations_path_list,
                                                   borders_tables_list):
    prob_locs_list = []
    for probe_locations_path in probe_locations_path_list:
        probe_locs = read_probe_mat(probe_locations_path)
        prob_locs = np.array([[probe_locs[0,0], probe_locs[0,1]], 
                              [probe_locs[2,0], probe_locs[2,1]],
                              [probe_locs[1,0], probe_locs[1,1]]])*10
        prob_locs_list.append(prob_locs)
    prob_locs_list = np.array(prob_locs_list)
    # adjust all probes to deepest probe depth
    max_y = np.max(prob_locs_list[:,1,1])
    for prob_locs, borders_table_path in zip(prob_locs_list, borders_tables_list):
        borders_table = pd.read_csv(borders_table_path)
        difference = max_y - prob_locs[1,1]
        # add additional row with a root name and with the difference 
        diff_row = pd.DataFrame({'upperBorder': [borders_table['lowerBorder'].iloc[-1]], 
                                 'lowerBorder': [borders_table['lowerBorder'].iloc[-1]+difference],
                                 'acronym': ['root'],
                                 'name': ['root'],
                                 'avIndex': [1]})
        borders_table = pd.concat([borders_table, diff_row], ignore_index=True)
        borders_table.to_csv(borders_table_path)
    return


def add_probe_SHARPTrack(probe_locations_path_list, color=None):
    cmap = ['darkgreen', 'green', 'limegreen', 'lime']

    prob_locs_list = []
    for probe_locations_path in probe_locations_path_list:
        probe_locs = read_probe_mat(probe_locations_path)
        radius=20
        prob_locs = np.array([[probe_locs[0,0], probe_locs[0,1]], 
                              [probe_locs[2,0], probe_locs[2,1]],
                              [probe_locs[1,0], probe_locs[1,1]]])*10
        prob_locs_list.append(prob_locs)
    prob_locs_list = np.array(prob_locs_list)
    # adjust all probes to deepest probe depth and plot probes
    max_y = np.max(prob_locs_list[:,1,1])
    for i, prob_locs in enumerate(prob_locs_list):
        prob_locs[1,1] = max_y

        # correct for the probe being place on the left side of the brain by mistake
        prob_locs_SC = prob_locs.copy()
        prob_locs_CCF = prob_locs.copy()
        prob_locs_SC[:,0] = CCFToStereo(prob_locs[:,0])
        prob_locs_SC[:,1] = CCFToStereo(prob_locs[:,1])
        if prob_locs_SC[:,0][2] > 0:
            prob_locs_SC[:,0][2]*=-1
            prob_locs_SC[:,1][2]*=-1
            prob_locs_CCF[:,0] = StereoToCCF(prob_locs_SC[:,0])
            prob_locs_CCF[:,1] = StereoToCCF(prob_locs_SC[:,1])
        else:
            prob_locs_CCF[:,0] = StereoToCCF(prob_locs_SC[:,0])
            prob_locs_CCF[:,1] = StereoToCCF(prob_locs_SC[:,1])
        
        #original location before correction
        #actor = Cylinder2(prob_locs[:,0], prob_locs[:,1], scene.root, color=color, radius=radius)
        #scene.add(actor)
        if color is None:
            c = cmap[i]
        else:
            c = color
        actor = Cylinder2(prob_locs_CCF[:,0], prob_locs_CCF[:,1], scene.root, color=c, radius=radius)
        scene.add(actor)
    return


# Injection site for LEC targetting from Vandrey et al. 2022
# Coordinate are [anterior-posterior, superior-inferior, left-right
# 3.8 mm posterior, 4 mm lateral

# Bregma to 4 mm immediately below Bregma (sanity check)
Inj2 = StereoToCCF(np.array([0,0,0]))
Tar2 = StereoToCCF(np.array([0,5000,0]))
print(Inj2)
print(Tar2)
channel_length = 200

settings.SHADER_STYLE = "cartoon"  # other options: metallic, plastic, shiny, glossy, cartoon, default
settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
settings.SHOW_AXES = True  # shows/hides the ABA CCF axes from the image
#show_atlases()  # this will print a list of atlases that you could use

scene = Scene(root=False, inset=False, atlas_name="allen_mouse_10um")  # makes a scene instance

root = scene.add_brain_region("root", alpha=0.05, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
mec = scene.add_brain_region("ENTm", alpha=0.25, color=(106, 202,71), hemisphere="both", silhouette=True)
#vis = scene.add_brain_region("VIS", alpha=0.25, color=(67,142,137), hemisphere="right", silhouette=True)


for Mouse, c_mouse in zip(['M20',     'M21',   'M22',  'M25', 'M26',     'M27',  'M28',  'M29'], 
                          ["blue", "yellow", "green", "cyan", "magenta", "red", "pink", "grey"]):

    '''
    adjust_border_table_for_apparent_longest_probe(probe_locations_path_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_1.mat',
                                                                              f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_3.mat',
                                                                              f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_2.mat',
                                                                              f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_4.mat'], 
                                                    borders_tables_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_1.csv',
                                                                         f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_2.csv',
                                                                         f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_3.csv',
                                                                         f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_4.csv'])
    '''
    add_probe_SHARPTrack(probe_locations_path_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_1.mat',
                                                    f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_2.mat',
                                                    f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_3.mat',
                                                    f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_4.mat'],
                                                    color=c_mouse)

scene.render(zoom=1.2)
print("hello there")

"""
anim = Animation(scene, "./examples", "vid1", fmt="mp4")

# Specify camera position and zoom at some key frames
# each key frame defines the scene's state after n seconds have passed
anim.add_keyframe(0, camera="sagittal", zoom=1)
anim.add_keyframe(10, camera="top", zoom=1)

anim.make_video(elevation=0, roll=0, azimuth=0.1, duration=20, fps=30)


# Make a custom make frame function
def make_frame(scene, frame_number, *args, **kwargs):
    alpha = scene.root.alpha()
    if alpha < 0.5:
        scene.root.alpha(1)
    else:
        scene.root.alpha(0.2)

# Create an instance of video maker
vm = VideoMaker(scene, "./examples", name="vid1", fmt="mp4")
# make a video with the custom make frame function
# this just rotates the scene
render_dict = {"zoom": 2,
               "camera": "sagittal"}
vm.make_video(elevation=0, roll=0, azimuth=0.5, duration=60, fps=60, render_kwargs=render_dict)
print("hello")
"""
