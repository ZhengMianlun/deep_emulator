import os
from os import listdir, makedirs, system, rmdir
import torch
import torch.nn as nn
import numpy as numpy
from model import Graph_MLP
import data_loader
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import shuffle

def train(net, writer, batch=48, train_seq_num = 10, test_seq_num = 10, total_iter=10, data_path_root = "../../data/", out_weight_folder="../train/"):
    if not os.path.exists(out_weight_folder):
        os.makedirs(out_weight_folder)

    print ("####Initiate model Graph_MLP")

    if(torch.cuda.is_available()):
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    
    start_point = 0
    for iteration in range(total_iter+1 - start_point):
        
        if(iteration % 1 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 * pow(0.96, iteration+start_point)
            print ("\n######## iteration " +str(iteration+start_point)) 
            learning_rate = 0.0001 * pow(0.96, iteration+start_point)
            print ("   learning rate:" + str(learning_rate))   
            writer.add_scalar('LearningRate', learning_rate, iteration+start_point)
        
        ##prepare training data
        # topology varying mesh, so we need to process them separately
        train_loss = 0.0
        train_loss_num = 0
        train_data_path_root = os.path.join(data_path_root, "train")
        train_files = sorted([f for f in listdir(train_data_path_root) if f.startswith("motion_")])
        shuffle(train_files)
        for train_file in train_files:
            mesh_path_root = os.path.join(train_data_path_root, train_file)
            frame_num = len([f for f in listdir(mesh_path_root+"/1/") if f.startswith("x_")]) - 1
            mesh_dataset_train = data_loader.Mesh_Dataset(mesh_path_root, train_seq_num, frame_num)
            training_data = DataLoader(mesh_dataset_train, batch_size = batch, shuffle = True, num_workers = 2)
            for constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass, output_f in training_data:
                output_pred = net(constraint[0,:], dynamic_f, reference_f, adj_matrix[0,:,:], stiffness, mass)
                if(torch.cuda.is_available()):
                    output_f = output_f.cuda()
                loss = net.compute_graph_loss(output_pred, output_f, constraint[0, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.item()
                train_loss_num = train_loss_num + 1
        train_loss = train_loss / train_loss_num

        if(iteration%1 == 0):
            print ("   training loss:" + str(train_loss))

        ##prepare test data
        test_loss = 0.0
        test_loss_num = 1
        test_data_path_root = os.path.join(data_path_root, "test")
        test_files = sorted([f for f in listdir(test_data_path_root) if f.startswith("motion_")])
        shuffle(test_files)
        for test_file in test_files:
            mesh_path_root = os.path.join(test_data_path_root, test_file)
            print(mesh_path_root)
            frame_num = len([f for f in listdir(mesh_path_root+"/1/") if f.startswith("x_")]) - 1
            mesh_dataset_test = data_loader.Mesh_Dataset(mesh_path_root, test_seq_num, frame_num)
            test_data = DataLoader(mesh_dataset_test, batch_size = 12, shuffle = True, num_workers = 2)
            for constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass, output_f in test_data:
                output_pred = net(constraint[0,:], dynamic_f, reference_f, adj_matrix[0,:,:], stiffness, mass)
                if(torch.cuda.is_available()):
                    output_f = output_f.cuda()
                loss = net.compute_graph_loss(output_pred, output_f, constraint[0, :])
                test_loss = test_loss + loss.item()
                test_loss_num = test_loss_num + 1
        test_loss = test_loss / test_loss_num
        
        if(iteration%1 == 0):
            print ("   test loss:" + str(test_loss))

        writer.add_scalar('Loss/train', train_loss, iteration+start_point)
        writer.add_scalar('Loss/test', test_loss, iteration+start_point)


        if(iteration %1 == 0):
            path = out_weight_folder + "_%07d"%(iteration+start_point)+".weight"
            torch.save(net.state_dict(), path)

print ("################Training#####################")
net = Graph_MLP()
writer = SummaryWriter('./runs/')
train(net, writer, batch=128, train_seq_num = 7, test_seq_num=7, total_iter=100, 
        data_path_root = "../data/sphere_dataset/", out_weight_folder="./weight/")
