import torch
from torch_geometric.nn import MessagePassing

class Interaction(MessagePassing):
    def __init__(self, in_edge, in_node, out_edge, out_node):
        super(Interaction, self).__init__(
            aggr='add',
            flow="source_to_target")
        self.in_edge = 2 * in_node + in_edge
        self.in_node = in_node + out_edge
        self.mlp_edge = torch.nn.Sequential(
            torch.nn.Linear(self.in_edge, out_edge, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_edge, out_edge, bias=True)
        )
        self.mlp_node = torch.nn.Sequential(
            torch.nn.Linear(self.in_node, out_node, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(out_node, out_node, bias=True)
        )
        
    def forward(self, x, edge_index, edge_attrs):
        return self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attrs=edge_attrs
        )
    
    def message(self, x_i, x_j, edge_index, edge_attrs):
        recv_send = [x_i, x_j]
        if edge_attrs is not None:
            recv_send.append(edge_attrs)
        recv_send = torch.cat(recv_send, dim=1)
        self.edge_embed = self.mlp_edge(recv_send)
        return self.edge_embed
    
    def update(self, aggr_out, x):
        node_embed = self.mlp_node(torch.cat([x, aggr_out], dim=1))
        return (self.edge_embed, node_embed)


class Net(torch.nn.Module):
    def __init__(self, dim_node=4, dim_edge=0, dim_embed_edge=64, dim_embed_node=16):
        super(Net, self).__init__()
        dim_embed_edge = 4
        self.conv1 = Interaction(dim_edge, dim_node,
                                 dim_embed_edge, dim_embed_node)
        dim_embed_edge = 8
        self.conv2 = Interaction(dim_embed_edge, dim_embed_node,
                                 dim_embed_edge, dim_embed_node)
        dim_embed_edge = 16
        self.conv3 = Interaction(dim_embed_edge, dim_embed_node,
                                 dim_embed_edge, dim_embed_node)
        dim_embed_edge = 32
        self.conv4 = Interaction(dim_embed_edge, dim_embed_node,
                                 dim_embed_edge, dim_embed_node)
        dim_embed_edge = 64
        self.conv5 = Interaction(dim_embed_edge, dim_embed_node,
                                 dim_embed_edge, dim_embed_node)
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(dim_embed_edge, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, data):
        node_attrs = data.x
        edge_attrs = data.edge_attr
        
        edge_attrs, node_attrs = self.conv1(node_attrs, data.edge_index, edge_attrs)
        edge_attrs, node_attrs = self.conv2(node_attrs, data.edge_index, edge_attrs)
        edge_attrs, node_attrs = self.conv3(node_attrs, data.edge_index, edge_attrs)
        edge_attrs, node_attrs = self.conv4(node_attrs, data.edge_index, edge_attrs)
        edge_attrs, node_attrs = self.conv5(node_attrs, data.edge_index, edge_attrs)
        
        return self.classify(edge_attrs)

