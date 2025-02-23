import os
import math
import pandas as pd
import spikeinterface.full as si
# read the matlab files
import numpy as np
import scipy.io
import pandas as pd
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
from brainrender.actors import Points
import matplotlib.pyplot as plt
from tifffile import imread

reference_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/reference.tiff')
annotations_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/annotation.tiff')
structure_set = pd.read_csv('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/structures.csv')

def read_probe_mat(probe_locs_path):
    mat = scipy.io.loadmat(probe_locs_path)
    probe_locs = np.array(mat['probe_locs'])
    print(probe_locs)
    return probe_locs


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

def reconstruct_shank_id(clusters_df):
    shank_ids = []
    for index, cluster in clusters_df.iterrows():
        x_pos = cluster['unit_location_x']
        if x_pos <= 150:
            shank_id = 0
        elif (x_pos > 150 and x_pos <= 400):
            shank_id = 1
        elif (x_pos > 400 and x_pos <= 650):
            shank_id = 2
        elif x_pos > 650:
            shank_id = 3
        shank_ids.append(shank_id)
    clusters_df['shank_id'] = shank_ids
    return clusters_df

def add_clusters(probe_locations_path_list, 
                 probe_borders_table_path_list,
                 clusters_df_path,
                 mouse):
    
    prob_locs_list = []
    borders_table_list = []
    for probe_locations_path, probe_borders_table_path in zip(probe_locations_path_list, probe_borders_table_path_list):
        probe_locs = read_probe_mat(probe_locations_path)
        borders_table = pd.read_csv(probe_borders_table_path)
        prob_locs = np.array([[probe_locs[0,0], probe_locs[0,1]], 
                              [probe_locs[2,0], probe_locs[2,1]],
                              [probe_locs[1,0], probe_locs[1,1]]])*10
        prob_locs_list.append(prob_locs)
        borders_table_list.append(borders_table)
    prob_locs_list = np.array(prob_locs_list)

    # adjust all probes to deepest probe depth
    max_y = np.max(prob_locs_list[:,1,1])
    for i in range(len(prob_locs_list)):
        prob_locs_list[i][1,1] = max_y
        prob_locs = prob_locs_list[i]

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
            continue
        prob_locs_list[i] = prob_locs_CCF
    
    
    # load dataframe
    clusters_df = pd.read_csv(clusters_df_path)

    # ignore cells with weird locations
    clusters_df = clusters_df[clusters_df['unit_location_x'] > -10]
    clusters_df = clusters_df[clusters_df['unit_location_y'] > -10]

    # remake shank_id parameter because it doesn't look very consistent. 
    clusters_df = reconstruct_shank_id(clusters_df)

    # create a x shift across the probe
    delta_xs = []
    for index, cluster in clusters_df.iterrows():
        shank_id = int(cluster['shank_id'])
        x_pos = cluster['unit_location_x']
        delta_x = np.nanmedian(clusters_df[clusters_df['shank_id'] == shank_id]['unit_location_x']) - x_pos # 250 microns is the assumed distance between NP2 probes
        delta_xs.append(delta_x)
    clusters_df['delta_x_um'] = delta_xs


    # unit_location_x is ML 
    # unit_location_y is DV
    # (0, 0) is the tip of the medial most shank and move +/+ in a lateral/dorsal direction
    # Add to scene

    # generate locations and annotations for a give cluster_dataframe which has columns
    # unit_location_x, unit_location_y, delta_x_um
    clusters_X_CCF = []
    clusters_Y_CCF = []
    clusters_Z_CCF = []
    clusters_X_SC = []
    clusters_Y_SC = []
    clusters_Z_SC = []
    clusters_annotations = []
    clusters_annotations_alternative = []

    for index, cluster in clusters_df.iterrows():
        shank_id = int(cluster['shank_id'])
        x_pos = cluster['unit_location_x']
        y_pos = cluster['unit_location_y']
        delta_x = cluster['delta_x_um']
        
        # get probe locations using shank id
        prob_locs_CCF = prob_locs_list[shank_id]
        borders_table = borders_table_list[shank_id]
        # translate to SC
        print('shank_id', shank_id)
        print('prob_locs_CCF', prob_locs_CCF)
         
        prob_locs_SC = prob_locs_CCF.copy()
        prob_locs_SC[:,0] = CCFToStereo(prob_locs_CCF[:,0])
        prob_locs_SC[:,1] = CCFToStereo(prob_locs_CCF[:,1])
        print('probe_length', euclidean_distance(prob_locs_SC[:,0], prob_locs_SC[:,1]))

        # Calculate the direction vector from P2 to P1
        direction_vector = prob_locs_SC[:,0] - prob_locs_SC[:,1]
        # Calculate the unit vector in the direction of P1 to P2
        unit_vector = direction_vector / np.linalg.norm(direction_vector)
        print('unit vector,', unit_vector)
        # Calculate the position P3, which is Y1 away from P2 along the direction of the unit vector
        cluster_pos_SC = prob_locs_SC[:,1] + (y_pos * unit_vector)
        cluster_pos_SC[2] += delta_x
        cluster_pos_CCF = StereoToCCF(cluster_pos_SC)

        # Calculate the euclidean distance between the probe tip and the cluster
        # then attribute the distance to an annotation in the border table
        cluster_distance_from_tip_CCF = euclidean_distance(cluster_pos_CCF, prob_locs_CCF[:,1])
        cluster_position_in_border_table = borders_table.iloc[-1]['lowerBorder']-cluster_distance_from_tip_CCF
        border_region = borders_table[(borders_table['upperBorder']<=cluster_position_in_border_table) &
                                      (borders_table['lowerBorder']>cluster_position_in_border_table)]
        if len(border_region) == 1:
            cluster_annotation = border_region['acronym'].iloc[0]
            clusters_annotations.append(cluster_annotation)

        # alternative way to find the location 
        # use the CCF coordinates, get an index and look up the annotation in the brainrender allen_brain_10um volume
        z_CCF,y_CCF,x_CCF = np.round(cluster_pos_CCF/10).astype(int)
        cluster_annotation_index = annotations_set[z_CCF,y_CCF,x_CCF]
        if len(structure_set[structure_set['id'] == cluster_annotation_index]) ==1:
            cluster_annotation = structure_set[structure_set['id'] == cluster_annotation_index]['acronym'].iloc[0]
            clusters_annotations_alternative.append(cluster_annotation)
        else: # assume out of brain?
            clusters_annotations_alternative.append('root')

        # add the estimated cluster location in the CCF brain
        scene.add(Points(np.reshape(cluster_pos_CCF, (1,3)), name=mouse, radius=30, colors="red"))

        clusters_X_CCF.append(cluster_pos_CCF[2]) 
        clusters_Y_CCF.append(cluster_pos_CCF[1])
        clusters_Z_CCF.append(cluster_pos_CCF[0])
        clusters_X_SC.append(cluster_pos_SC[2])
        clusters_Y_SC.append(cluster_pos_SC[1])
        clusters_Z_SC.append(cluster_pos_SC[0])
    
    clusters_df['X_CCF'] = clusters_X_CCF
    clusters_df['Y_CCF'] = clusters_Y_CCF
    clusters_df['Z_CCF'] = clusters_Z_CCF
    clusters_df['X_SC'] = clusters_X_SC
    clusters_df['Y_SC'] = clusters_Y_SC
    clusters_df['Z_SC'] = clusters_Z_SC
    clusters_df['cluster_annotation_derived_from_SHARP-Track'] = clusters_annotations
    clusters_df['cluster_annotation_derived_from_brain_render'] = clusters_annotations_alternative
    clusters_df.to_csv('/Users/harryclark/Documents/probe_data/'+mouse+'_corrected_with_annotation.csv')
    return clusters_df
    

def euclidean_distance(point1, point2):
    distance = np.sqrt((point2[0] - point1[0])**2 +
                       (point2[1] - point1[1])**2 +
                       (point2[2] - point1[2])**2)
    return distance

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
#root = scene.add_brain_region("root", alpha=1, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
root = scene.add_brain_region("root", alpha=0.05, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
mec = scene.add_brain_region("ENTm", alpha=0.25, color=(106, 202,71), hemisphere="both", silhouette=True)
#vis = scene.add_brain_region("VIS", alpha=0.25, color=(67,142,137), hemisphere="right", silhouette=True)
#for Mouse, c_mouse in zip(['M20', 'M21', 'M22', 'M25', 'M26', 'M27', 'M28', 'M29'], 
#                          ["orange", "blue", "yellow", "green", "cyan", "magenta", "red", "pink", "grey"]):

#actor = Cylinder2(np.array([0,0,0]), np.array([0,8000,0]),scene.root, color='red', radius=20)
#scene.add(actor)
#actor = Cylinder2(np.array([0,0,0]), np.array([0,0, 11400]),scene.root, color='blue', radius=20)
#scene.add(actor)
#actor = Cylinder2(np.array([0,0,0]), np.array([13200,0,0]),scene.root, color='green', radius=20)
#scene.add(actor)

for Mouse in ['M25']: 
    clusters_df_path = f'/Users/harryclark/Documents/probe_data/{Mouse}.csv'

    add_probe_SHARPTrack(probe_locations_path_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_1.mat',
                                                f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_2.mat',
                                                f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_3.mat',
                                                f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_4.mat'], color=None)

for Mouse in ['M25']:
    clusters_df = add_clusters(probe_locations_path_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_1.mat',
                                                          f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_2.mat',
                                                          f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_3.mat',
                                                          f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_locations_4.mat'],
                                probe_borders_table_path_list=[f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_1.csv',
                                                               f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_2.csv',
                                                               f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_3.csv',
                                                               f'/Users/harryclark/Documents/probe_data/{Mouse}_probe_border_table_4.csv'],
                               clusters_df_path=clusters_df_path,
                               mouse=Mouse)
    
scene.render(zoom=1.2)  
print("stoppp!")   