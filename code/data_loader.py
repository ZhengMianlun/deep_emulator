'''
Author: Mianlun Zheng
load data for network training and testing
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

# form the dataset for batch training 
class Mesh_Dataset(Dataset):
    def __init__(self, mesh_path_root, seq_num, frame_num):
        self.constraint, self.dynamic_f, self.reference_f, self.adj_matrix, self.stiffness, self.mass = loadTrainingInputData(mesh_path_root, seq_num, frame_num)
        self.output_f = loadTrainingOutputData(mesh_path_root, seq_num, frame_num)
        self.sample_num = self.output_f.shape[0]

    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, idx):
        return self.constraint, self.dynamic_f[idx], self.reference_f[idx], self.adj_matrix, self.stiffness[idx], self.mass[idx], self.output_f[idx]

# data formats:
# u_i, the dynamic mesh at frame i, vertices# * 3
# x_i, the reference mesh at frame i, vertices# * 3
# c, the unconstraint information, vertices# * 1
# m, the mass, vertices# * 1
# k, the stiffness, 1
# adj, the adjacency matrix, #vertices# * max_neighbors#
#      !!! note !!!: to keep the shape regular, we filled the blank positions with 0
#      !!! note !!!: 1-indexed
#       eg: [ [1,2,3], [1,0,0] ], 
#       so, the first columns of the input feature should be filled with 0

# training data formats:
# input data: 
# unconstraint: batch * vertex_num * 1
# +1 dimension: filled with 0 values in case the vertex's negibors < max_neighbors#
# dynamic_f: batch * (vertex_num+1) * 9
#            u_i(+0), u_i(-1), u_i(-2) 
# reference_f: batch * (vertex_num+1) * 9
#            x_i(+0), x_i(-1), x_i(-2) 
# adj_matrix: batch * vertex_num * max_neighbors#
# mass: (vertex_num+1) * 1
# stiffness: batch * vertex_num * max_neighbors#
# stiffness/mass: batch * vertex_num * 1
# output data:
# feature: displacements at a certain vertex: x, y, z
# vertcies# * 3

# load binary file of float data
def loadData_Float(filename):
    data = np.fromfile(filename, dtype=np.float64)
    data = np.array(data)
    return data

# load binary file of int data
def loadData_Int(filename):
    data = np.fromfile(filename, dtype=np.int32)
    data = np.array(data)
    return data

# load input data for training
def loadTrainingInputData(file_path, seq_num, frame_num):
    dynamic_f = []
    reference_f = []
    stiffness = []
    mass = []
    for i in range(seq_num):
        for j in range(frame_num-1):
            # current dynamic mesh u(0)
            for k in range(-2, 1):
                u_curr_filename = ""
                if(j+k<0):
                    u_curr_filename = os.path.join(file_path, str(i+1), "u_" + str(0))
                elif(j+k>frame_num-1):
                    u_curr_filename = os.path.join(file_path, str(i+1), "u_" + str(frame_num-1))
                else:
                    u_curr_filename = os.path.join(file_path, str(i+1), "u_" + str(j+k))
                u_curr = loadData_Float(u_curr_filename).reshape((-1,3))
                u_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), u_curr), 0)
                if(k==-2):
                    u = u_curr
                else:
                    u = np.concatenate((u_curr, u), axis = 1)
            for k in range (-1, 2):
                x_curr_filename = ""
                if(j+k<0):
                    x_curr_filename = os.path.join(file_path, str(i+1), "x_" + str(0))
                elif(j+k>frame_num-1):
                    x_curr_filename = os.path.join(file_path, str(i+1), "x_" + str(frame_num-1))
                else:
                    x_curr_filename = os.path.join(file_path, str(i+1), "x_" + str(j+k))
                x_curr = loadData_Float(x_curr_filename).reshape((-1,3))
                x_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), x_curr), 0)
                if(k==-1):
                    x = x_curr
                else:
                    x = np.concatenate((x_curr, x), axis = 1)

            k_filename = os.path.join(file_path, str(i+1), "k") # stiffness for each vertex
            k = loadData_Float(k_filename) 
            k = np.expand_dims(k, axis=1)
            k = k * 0.000001
            m_filename = os.path.join(file_path, str(i+1), "m") # mass for each vertex
            m = loadData_Float(m_filename) 
            m = np.expand_dims(m, axis=1)
            m = m * 1000
            m[0] = 1.0

            dynamic_f.append(u)
            reference_f.append(x)
            stiffness.append(k)
            mass.append(m)

    c_filename = os.path.join(file_path, "1/c") # constraint information as mask, 0/1: unconstrained / constrained; same over the same mesh
    constraint = loadData_Int(c_filename)

    adj_filename = os.path.join(file_path, "1/adj") # adjacent information for each vertex
    adj_matrix = loadData_Int(adj_filename).reshape((u.shape[0], -1))
    
    constraint = torch.from_numpy(np.array(constraint)).type(torch.LongTensor)
    dynamic_f = torch.from_numpy(np.array(dynamic_f)).float()
    reference_f = torch.from_numpy(np.array(reference_f)).float()
    adj_matrix = torch.from_numpy(np.array(adj_matrix)).type(torch.LongTensor)
    stiffness = torch.from_numpy(np.array(stiffness)).float()
    mass = torch.from_numpy(np.array(mass)).float()

    return constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass

# form the output features for the whole graph
# feature: displacements at a certain vertex: x, y, z
# vertcies# * 3
def loadTrainingOutputData(file_path, seq_num, frame_num):
    data = []
    for i in range(seq_num):
        for j in range(frame_num-1):
            u_next_filename = os.path.join(file_path, str(i+1), "u_" + str(j+1))
            u_next = loadData_Float(u_next_filename).reshape((-1,3))
            u_next = np.concatenate((np.zeros((1,3), dtype = np.float64), u_next), 0)
            u_curr_filename = os.path.join(file_path, str(i+1), "u_" + str(j))
            u_curr = loadData_Float(u_curr_filename).reshape((-1,3))
            u_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), u_curr), 0)
            u = u_next - u_curr
            data.append(u)
    data = torch.from_numpy(np.array(data)).float()
    return data

def loadTestInputData(file_path, curr_frame, frame_num):
    dynamic_f = []
    reference_f = []
    stiffness = []
    mass = []
    # current dynamic mesh u(0)
    for k in range(-2, 1):
        u_curr_filename = ""
        if(curr_frame+k<0):
            u_curr_filename = os.path.join(file_path, "u_0")
        elif(curr_frame+k>frame_num-1):
            u_curr_filename = os.path.join(file_path, "u_" + str(frame_num-1))
        else:
            u_curr_filename = os.path.join(file_path, "u_" + str(curr_frame+k))
        u_curr = loadData_Float(u_curr_filename).reshape((-1,3))
        u_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), u_curr), 0)
        if(k==-2):
            u = u_curr
        else:
            u = np.concatenate((u_curr, u), axis = 1)
    
    for k in range (-1, 2):
        x_curr_filename = ""
        if(curr_frame+k<0):
            x_curr_filename = os.path.join(file_path, "x_0")
        elif(curr_frame+k>frame_num-1):
            x_curr_filename = os.path.join(file_path, "x_" + str(frame_num-1))
        else:
            x_curr_filename = os.path.join(file_path, "x_" + str(curr_frame+k))
        x_curr = loadData_Float(x_curr_filename).reshape((-1,3))
        x_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), x_curr), 0)
        if(k==-1):
            x = x_curr
        else:
            x = np.concatenate((x_curr, x), axis = 1)
    
    c_filename = os.path.join(file_path, "c")
    constraint = loadData_Int(c_filename)
    adj_filename = os.path.join(file_path, "adj") # adjacent information for each vertex
    adj_matrix = loadData_Int(adj_filename).reshape((u.shape[0], -1))

    k_filename = os.path.join(file_path, "k") # stiffness for each vertex
    k = loadData_Float(k_filename) 
    k = np.expand_dims(k, axis=1)
    k = k * 0.000001
    m_filename = os.path.join(file_path, "m") # mass for each vertex
    m = loadData_Float(m_filename) 
    m = np.expand_dims(m, axis=1)
    m = m * 1000
    m[0] = 1.0
    
    dynamic_f.append(u)
    reference_f.append(x)
    stiffness.append(k)
    mass.append(m)

    constraint = torch.from_numpy(np.array(constraint)).type(torch.LongTensor)
    dynamic_f = torch.from_numpy(np.array(dynamic_f)).float()
    reference_f = torch.from_numpy(np.array(reference_f)).float()
    adj_matrix = torch.from_numpy(np.array(adj_matrix)).type(torch.LongTensor)
    stiffness = torch.from_numpy(np.array(stiffness)).float()
    mass = torch.from_numpy(np.array(mass)).float()

    return constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass

# form the output features for the whole graph
# feature: displacements at a certain vertex: x, y, z
# vertcies# * 3
def loadTestOutputData(file_path, curr_frame, frame_num):
    output_f = []
    u_next_filename = os.path.join(file_path, "u_" + str(curr_frame+1))
    u_next = loadData_Float(u_next_filename).reshape((-1,3))
    u_next = np.concatenate((np.zeros((1,3), dtype = np.float64), u_next), 0)
    u_curr_filename = os.path.join(file_path, "u_" + str(curr_frame))
    u_curr = loadData_Float(u_curr_filename).reshape((-1,3))
    u_curr = np.concatenate((np.zeros((1,3), dtype = np.float64), u_curr), 0)
    u = u_next - u_curr
    output_f.append(u)
    output_f = torch.from_numpy(np.array(output_f)).float()
    return output_f