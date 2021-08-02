import torch
import torch.nn as nn
import numpy as np

# MLP for a local patch of the surface mesh
class Graph_MLP(nn.Module):
    def  __init__(self):
        super(Graph_MLP, self).__init__()
   
        # mlp beta
        self.edge_mlp = nn.Sequential(
            nn.Linear(37, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        # mlp alpha
        self.point_mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64)
        )

        # mlp gamma
        self.instance_mlp = nn.Sequential(
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 3)
        )

    def form_feature(self, constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass):
        
        # form the mask
        mask = torch.ones(adj_matrix.shape[0], adj_matrix.shape[1])
        mask[adj_matrix == 0] = 0.0
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(3)
        mask = mask.expand(dynamic_f.shape[0], -1, -1, -1)

        # form the feature on the center point
        # batch * vertex_num * 15
        point_f_u_curr = dynamic_f[:, :, 0:3] - reference_f[:, :, 3:6]
        point_f_u_pre = dynamic_f[:, :, 3:6] - reference_f[:, :, 3:6]
        point_f_u_ppre = dynamic_f[:, :, 6:9] - reference_f[:, :, 3:6]
        point_f_x_next = reference_f[:, :, 0:3] - reference_f[:, :, 3:6]
        point_f_x_pre = reference_f[:, :, 6:9] - reference_f[:, :, 3:6]
        point_f_k = stiffness
        point_f_k_m = stiffness / mass
        point_f = torch.cat((point_f_u_curr, point_f_u_pre, point_f_u_ppre, point_f_x_next, point_f_x_pre, point_f_k, mass, point_f_k_m), 2)
        
        # update the information on the constrained points
        # dynamic_f[:, constraint==0, 0:3] = reference_f[:, constraint==0, 0:3]
        # form the feature on the neighbor
        # batch * vertex_num * max_neighbor * 6
        edge_f_c = constraint[adj_matrix]
        edge_f_c = edge_f_c.unsqueeze(0)
        edge_f_c = edge_f_c.unsqueeze(3)
        edge_f_c = edge_f_c.expand(dynamic_f.shape[0], -1, -1, -1)

        neighbor_f_u_curr = dynamic_f[:, adj_matrix, 0:3] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f_u_pre = dynamic_f[:, adj_matrix, 3:6] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f_u_ppre = dynamic_f[:, adj_matrix, 6:9] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f_x_next = reference_f[:, adj_matrix, 0:3] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f_x_curr = reference_f[:, adj_matrix, 3:6] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f_x_pre = reference_f[:, adj_matrix, 6:9] - reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        neighbor_f = torch.cat((neighbor_f_u_curr, neighbor_f_u_pre, neighbor_f_u_ppre, neighbor_f_x_next, neighbor_f_x_curr, neighbor_f_x_pre), 3)

        edge_f = torch.cat((edge_f_c.float(), neighbor_f, point_f.unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)), 3)       
        
        point_f = point_f[:, constraint==0, :]
        edge_f = edge_f[:, constraint==0, :, :]
        mask = mask[:, constraint==0, :, :]

        if(torch.cuda.is_available()):
            return edge_f.cuda(), point_f.cuda(), mask.cuda()
        else:
            return edge_f, point_f, mask


    def forward(self, constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass):
        # input data:
        # unconstraint: batch * (vertex_num + 1)* 1
        # +1 dimension: filled with 0 values in case the vertex's negibors < max_neighbors#
        # dynamic_f: batch * (vertex_num+1) * 9
        #            u_i(+0), u_i(-1), u_i(-2) 
        # reference_f: batch * (vertex_num+1) * 9
        #            x_i(+0), x_i(-1), x_i(-2) 
        # adj_matrix: batch * vertex_num * max_neighbors#
        # mass: (vertex_num+1) * 1
        # stiffness: batch * (vertex_num +1 )* max_neighbors#
        
        edge_f, point_f, mask = self.form_feature(constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass)   
        
        output_edge = self.edge_mlp(edge_f)  
        output_edge = output_edge * mask
        output_edge = torch.sum(output_edge, 2)

        output_point = self.point_mlp(point_f)

        input_instance = torch.cat((output_point, output_edge), 2)
        output_instance = self.instance_mlp(input_instance)

        return output_instance

    def compute_graph_loss(self, output_pred, output_f, constraint):
        output_f = output_f[:, constraint==0, :]
        loss = torch.pow(output_pred-output_f, 2).mean()
        if(torch.cuda.is_available()):
            return loss.cuda()
        else:
            return loss