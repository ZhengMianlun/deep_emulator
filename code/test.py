import os, glob
import torch
import torch.nn as nn
import numpy as np
from model import Graph_MLP
import data_loader
from torch.utils.data import Dataset, DataLoader
import render
import trimesh
import cv2 as cv
from matplotlib import cm
import time
import subprocess
from os import listdir, makedirs, system, rmdir
import animationTet2Surface
import sys

def loadData_Float(filename):
    data = np.fromfile(filename, dtype=np.float64)
    data = np.array(data)
    data = data[1:]
    data = np.reshape(data, (-1,3))
    return data

def predict_rollout(net, frame_num, mesh_path_root, data_path_root, eval_path_root, flag, character_name = None, camera_set = [6.0, 2.0, 5.0, 0.3]):
    if not os.path.exists(eval_path_root):
        os.makedirs(eval_path_root)
    
    net.eval()
    if(torch.cuda.is_available()):
        net = net.cuda()
    
    u_curr = 0
    u_pre = 0
    u_ppre = 0
    
    offset_filename = os.path.join(data_path_root, "offset")
    offset = data_loader.loadData_Float(offset_filename)
    offset = np.reshape(offset, (-1, 3))
        
    for k in range(frame_num-1):
        constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass = data_loader.loadTestInputData(data_path_root, k, frame_num)
        output_f = data_loader.loadTestOutputData(data_path_root, k, frame_num)
        dis_true = dynamic_f[:,:,0:3] + output_f
        dis_input = reference_f[:, :, 3:6] 

        if(k == 0):
                u_curr = dynamic_f[:, constraint==0, 0:3] 
                u_pre = dynamic_f[:, constraint==0, 3:6]
                u_ppre = dynamic_f[:, constraint==0, 6:9] 
        else:
                dynamic_f[:, constraint==0, 0:3] = u_curr
                dynamic_f[:, constraint==0, 3:6] = u_pre
                dynamic_f[:, constraint==0, 6:9] = u_ppre
                    
        output_pred = net(constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass).cpu().data
        dis_pred = torch.zeros(1, dynamic_f.shape[1], 3)
        dis_pred[:, constraint==0, :] = dynamic_f[:,constraint==0,0:3]+output_pred
        dis_pred[:, constraint==1, :] = reference_f[:,constraint==1,3:6]+output_f[:, constraint==1, :]
                    
        u_ppre = u_pre
        u_pre = u_curr
        u_curr = dis_pred[:, constraint==0, :]

        # render the prediction in tet mesh    
        if("tet" in flag):
            mesh_filename = os.path.join(mesh_path_root, "rest.ply")
            mesh_true = trimesh.load(mesh_filename, process = False)
            mesh_pred = trimesh.load(mesh_filename, process = False)
            mesh_input = trimesh.load(mesh_filename, process = False)

            for v, vertex in enumerate(mesh_input.vertices):
                mesh_input.vertices[v] = vertex + np.array(dis_input[:, v+1, :]) - offset[v, :]

            for v, vertex in enumerate(mesh_true.vertices):
                mesh_true.vertices[v] = vertex + np.array(dis_true[:, v+1, :]) - offset[v, :]
                        
            for v, vertex in enumerate(mesh_pred.vertices):
                mesh_pred.vertices[v] = vertex + np.array(dis_pred[:, v+1, :]) - offset[v, :]

            for v, vertex in enumerate(mesh_true.vertices):
                color = [0.0, 0.0, 0.0, 255.0]
                if(stiffness[0,v+1,:] == 5.0):
                    color = [64, 64, 64, 255.0]
                if(stiffness[0,v+1,:] == 1.0):
                    color = [255, 51, 153, 255.0]
                if(stiffness[0,v+1,:] == 0.5):
                    color = [153, 82, 255, 255.0]
                if(stiffness[0,v+1,:] == 0.1):
                    color = [51, 51, 255, 255.0]
                if(stiffness[0,v+1,:] == 0.05):
                    color = [59, 76, 192, 255.0]
                mesh_input.visual.vertex_colors[v] = color 
                mesh_true.visual.vertex_colors[v] = color 
                mesh_pred.visual.vertex_colors[v] = color 

            for i, c in enumerate(constraint):
                if(c == 1 and i >0):
                    mesh_input.visual.vertex_colors[i-1] = [255, 0, 0, 255.0]    
                    mesh_true.visual.vertex_colors[i-1] = [255, 0, 0, 255.0]    
                    mesh_pred.visual.vertex_colors[i-1] = [255, 0, 0, 255.0]   

            image_input = render.render_single_mesh(mesh_input, camera_set = camera_set, resolution=[500,800])
            image_true = render.render_single_mesh(mesh_true, camera_set = camera_set, resolution=[500,800])
            image_pred = render.render_single_mesh(mesh_pred, camera_set = camera_set, resolution=[500,800])
            image = np.concatenate((image_input, image_true, image_pred), axis=1)
            img_filename = os.path.join(eval_path_root, str(k) + ".png")
            print(img_filename)
            cv.imwrite(img_filename,image)
        
        # save the prediction to surface animation
        elif("surface" in flag):
            surfaceMesh_filename = os.path.join(mesh_path_root, "surface_render.ply")
            surfaceMesh_input = trimesh.load(surfaceMesh_filename, process = False)
            surfaceMesh_true = trimesh.load(surfaceMesh_filename, process = False)
            surfaceMesh_pred = trimesh.load(surfaceMesh_filename, process = False)

            pred_dis = dis_pred[:, 1:, :]- offset
            pred_dis = np.reshape(pred_dis, (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "pre_u"), pred_dis)

            true_dis = dis_true[:, 1:, :]- offset
            true_dis = np.reshape(true_dis, (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "true_u"), true_dis)
            
            input_dis = dis_input[:, 1:, :]- offset
            input_dis = np.reshape(input_dis, (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "input_u"), input_dis)

            animationTet2Surface.animationTet2Surface(mesh_path_root = mesh_path_root, eval_path_root = eval_path_root, character_name = character_name, prefix = "pre_")
            animationTet2Surface.animationTet2Surface(mesh_path_root = mesh_path_root, eval_path_root = eval_path_root, character_name = character_name, prefix = "input_")
            animationTet2Surface.animationTet2Surface(mesh_path_root = mesh_path_root, eval_path_root = eval_path_root, character_name = character_name, prefix = "true_")

            surface_input_dis = loadData_Float(os.path.join(eval_path_root, "input_SurfaceDis.u"))
            surface_true_dis = loadData_Float(os.path.join(eval_path_root, "true_SurfaceDis.u"))
            surface_pre_dis = loadData_Float(os.path.join(eval_path_root, "pre_SurfaceDis.u"))
        
            for v, vertex in enumerate(surfaceMesh_input.vertices):
                surfaceMesh_input.vertices[v] = vertex + surface_input_dis[v, :]

            for v, vertex in enumerate(surfaceMesh_true.vertices):
                surfaceMesh_true.vertices[v] = vertex + surface_true_dis[v, :]

            for v, vertex in enumerate(surfaceMesh_pred.vertices):
                surfaceMesh_pred.vertices[v] = vertex + surface_pre_dis[v, :]
            
            image_input = render.render_single_mesh(surfaceMesh_input, camera_set = camera_set, resolution=[500,800])
            image_true = render.render_single_mesh(surfaceMesh_true, camera_set = camera_set, resolution=[500,800])
            image_pred = render.render_single_mesh(surfaceMesh_pred, camera_set = camera_set, resolution=[500,800])
            image = np.concatenate((image_input, image_true, image_pred), axis=1)
            img_filename = os.path.join(eval_path_root, str(k) + ".png")
            print(img_filename)
            cv.imwrite(img_filename,image)


character_name = "michelle"
motion_name = "cross_jumps"
weight_path = "./weight/example.weight" 
mesh_path_root = "../data/character_dataset/"
eval_path_root = "./weight/eval/"
camera_set = [6.0, 2.0, 5.0, 0.3]

input_path = os.path.join(mesh_path_root, character_name, motion_name)
mesh_path = os.path.join(mesh_path_root, character_name)
frames = len([f for f in listdir(input_path) if f.startswith("x_")])
eval_path = os.path.join(eval_path_root, character_name, motion_name)

net = Graph_MLP()
net.load_state_dict(torch.load(weight_path))
predict_rollout(net, frame_num = frames, mesh_path_root = mesh_path, data_path_root = input_path, eval_path_root = eval_path, flag = "surface", character_name = character_name, camera_set = camera_set)