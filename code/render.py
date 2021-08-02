"""using pyrender for viewing and offscreen rendering.
"""
import numpy as np
import os
import trimesh
import pyrender
import matplotlib.pyplot as plt
from transforms3d import euler
import cv2 as cv

 

def get_transformation(r_axis, theta, T):
    # first set the camera postion
    # then rotate the camera
    R = euler.axangle2mat(r_axis,theta)
    camera_pose = np.zeros((4,4))
    camera_pose[0:3,0:3] = R
    camera_pose[0:3, 3] = np.array(T)
    camera_pose[3,3]=1  
    return camera_pose

#suppose model is centered at (0,0,0) 
def render_single_mesh(trimesh = None, enable_wire = False, camera_set = [6.0, 2.0, 5.0, 0.3], resolution = [400,800], enableAmbient=True, lightIntensity = 2):
    scene = pyrender.Scene()
    if enableAmbient:
        scene = pyrender.Scene(ambient_light = [1.0,1.0,1.0])
    if(trimesh):
        mesh_input = pyrender.Mesh.from_trimesh(trimesh, wireframe = False, smooth=False)
        scene.add(mesh_input)
        if enable_wire:
            for v, vertex in enumerate(trimesh.vertices):
                trimesh.visual.vertex_colors[v] = [0, 0, 0, 255.0] #
            mesh_input = pyrender.Mesh.from_trimesh(trimesh, wireframe = True, smooth=False)
            scene.add(mesh_input)   
    #camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, aspectRatio=resolution[0]/resolution[1])
    camera_pose = get_transformation(r_axis=[0,1,0], theta=np.pi*camera_set[3], T= [camera_set[0], camera_set[1], camera_set[2]])
    scene.add(camera, pose=camera_pose)
    
    #light
    key_light = pyrender.DirectionalLight(color = np.ones(3),intensity=lightIntensity)
    key_light_pose = get_transformation(r_axis=[1,1,0], theta=np.pi*-0.25, T= [-1.5,camera_set[1]+2.0,3.0])
    scene.add(key_light, pose=key_light_pose)
    fill_light = pyrender.DirectionalLight(color = np.ones(3), intensity=lightIntensity)
    fill_light_pose = get_transformation(r_axis=[0,1,0], theta=np.pi*0.25, T= [1.5,camera_set[1]-2.0,3.0])
    scene.add(fill_light, pose=fill_light_pose)
    back_light = pyrender.DirectionalLight(color = np.ones(3),intensity=lightIntensity)
    back_light_pose = get_transformation(r_axis=[-0.5,1,0], theta=np.pi*(1+0.25), T= [-1.5,camera_set[1]+2.0,3.0])
    scene.add(back_light, pose=back_light_pose)

    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    # render_flags = 16|8
    color, depth = r.render(scene)
    cv_color = np.ones(color.shape)
    cv_color[:,:,0]=color[:,:,2]
    cv_color[:,:,1]=color[:,:,1]
    cv_color[:,:,2]=color[:,:,0]

    return cv_color