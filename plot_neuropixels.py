# Example code to plot trajectories in BrainRender from stereotaxic coordinates

import math
import numpy as np

from brainrender import Scene
from brainrender.actors import Cylinder
from brainrender import VideoMaker
from brainrender import Animation
from brainrender import settings
from bg_atlasapi import show_atlases

# Extension to cylinder class to allow specification of start and end coordinates.
# Thanks to Ian Hawes for this.
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


def add_probes_in_x(cranial_site_xyz= np.array([0, 0, 0]), depth=1000,
               attack_angle_x=-15, color="red", distance_between_probes=250, radius=20, mark_true=False):
    # presumes the only angle that varies is the y axis of the stereotax (What we call Y but is called X here)

    attack_angle_x = attack_angle_x*-1
    attack_angle_x = np.deg2rad(attack_angle_x)
    z=0
    y = depth*np.cos(attack_angle_x);
    x = depth*np.sin(attack_angle_x);

    target_site = cranial_site_xyz + np.array([x,y,z])

    for i in range(4):
        if i == 0 and mark_true:
            c_color="red"
        else:
            c_color=color
        probe_offset = np.array([0,0, -i*distance_between_probes])
        actor = Cylinder2(StereoToCCF(cranial_site_xyz+probe_offset), StereoToCCF(target_site+probe_offset), scene.root, color=c_color, radius=radius)
        scene.add(actor)
    return

def add_probes_in_y(cranial_site_xyz= np.array([0, 0, 0]), depth=1000,
                    attack_angle_x=-15, color="red", distance_between_probes=250, radius=20, mark_true=False):
    # presumes the only angle that varies is the y axis of the stereotax (What we call Y but is called X here)

    attack_angle_x = attack_angle_x*-1
    attack_angle_x = np.deg2rad(attack_angle_x)
    z=0
    y = depth*np.cos(attack_angle_x);
    x = depth*np.sin(attack_angle_x);
    target_site = cranial_site_xyz + np.array([x,y,z])

    for i in range(4):
        if i == 0 and mark_true:
            c_color="red"
        else:
            c_color=color
        probe_offset = np.array([np.cos(attack_angle_x)*(-i*distance_between_probes),np.sin(attack_angle_x)*(i*distance_between_probes),0])
        actor = Cylinder2(StereoToCCF(cranial_site_xyz+probe_offset), StereoToCCF(target_site+probe_offset), scene.root, color=c_color, radius=radius)
        scene.add(actor)
    return

def add_neuropixel2_multishank_in_x(cranial_site_xyz, attack_angle_x, depth, radius=35, mark_true=False):
    add_probes_in_x(cranial_site_xyz= cranial_site_xyz, attack_angle_x=attack_angle_x, color="black", depth=depth, radius=radius)
    add_probes_in_x(cranial_site_xyz= cranial_site_xyz, attack_angle_x=attack_angle_x, color=(75,75,75), depth=depth-3000, radius=radius+1, mark_true=mark_true)
    return

def add_neuropixel2_multishank_in_y(cranial_site_xyz, attack_angle_x, depth, radius=35, mark_true=False):
    add_probes_in_y(cranial_site_xyz= cranial_site_xyz, attack_angle_x=attack_angle_x, color="black", depth=depth, radius=radius)
    add_probes_in_y(cranial_site_xyz= cranial_site_xyz, attack_angle_x=attack_angle_x, color=(75,75,75), depth=depth-0.01, radius=radius+1, mark_true=mark_true)
    return

# Injection site for LEC targetting from Vandrey et al. 2022
# Coordinate are [anterior-posterior, superior-inferior, left-right
# 3.8 mm posterior, 4 mm lateral

# Bregma to 4 mm immediately below Bregma (sanity check)
Inj2 = StereoToCCF(np.array([0,0,0]))
Tar2 = StereoToCCF(np.array([0,5000,0]))
channel_length = 200

settings.SHADER_STYLE = "cartoon"  # other options: metallic, plastic, shiny, glossy, cartoon, default
settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
settings.SHOW_AXES = False  # shows/hides the ABA CCF axes from the image
#show_atlases()  # this will print a list of atlases that you could use
scene = Scene(root=False, inset=False, atlas_name="allen_mouse_10um")  # makes a scene instance

root = scene.add_brain_region("root", alpha=0.05, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
mec = scene.add_brain_region("ENTm", alpha=0.25, color=(106, 202,71), hemisphere="right", silhouette=True)
#vis = scene.add_brain_region("VIS", alpha=0.25, color=(67,142,137), hemisphere="right", silhouette=True)

mec_cranial_site = np.array([3600,-1000,-3200])
mec_implant_depth = 5000
add_neuropixel2_multishank_in_x(cranial_site_xyz= mec_cranial_site,
                                attack_angle_x=-10, depth=mec_implant_depth, radius=35, mark_true=False)

# add a marker for bregma
#actor = Cylinder2(Inj2, Tar2, scene.root, color="black", radius=25)
#scene.add(actor)

scene.render(zoom=1.2)
print("hello there")
"""
anim = Animation(scene, "./examples", "vid1", fmt="mp4")

# Specify camera position and zoom at some key frames
# each key frame defines the scene's state after n seconds have passed
anim.add_keyframe(0, camera="sagittal", zoom=1)
anim.add_keyframe(10, camera="top", zoom=1)

anim.make_video(elevation=0, roll=0, azimuth=0.1, duration=20, fps=30)
"""
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
