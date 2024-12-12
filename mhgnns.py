import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GCNConv,global_mean_pool,TopKPooling
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.utils import negative_sampling
from tqdm import trange

import torch.nn as nn
from torch_geometric.nn import HeteroConv

from cmd_args import parse_args
from config import cfg,load_cfg,set_out_dir,dump_cfg


class MHGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MHGNN, self).__init__()
        
        # Message-passing layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Hierarchical pooling
        self.pool1 = TopKPooling(hidden_dim, ratio=0.8)
        
        # Fully connected layers for node classification
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)  # `num_classes` is the number of parent nodes
        
    def forward(self, x, edge_index, batch, node_level):
        # Initial graph convolution
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Apply pooling to capture hierarchical relationships
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch)
        
        # Use node-level information to focus on product nodes
        product_mask = (node_level == "product")  # Boolean mask for product nodes
        product_features = x[product_mask]  # Features only for product nodes
        
        # Fully connected layers for classification
        x = F.relu(self.fc1(product_features))
        x = self.fc2(x)  # Predict the parent class for each product
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels,heads=1):
        super().__init__()
        # HeteroConv combines layers for each edge type
        #('products','suppliedby','vendors'): SAGEConv(-1, hidden_channels),
        self.convs = HeteroConv({
            ('subcats','childof','cats'):SAGEConv(-1, hidden_channels),
            ('depts','childof','subcats'):SAGEConv(-1, hidden_channels),
            ('subdepts','childof','depts'):SAGEConv(-1, hidden_channels),
            ('classes','childof','subdepts'):SAGEConv(-1, hidden_channels),
            ('products','childof','classes'):SAGEConv(-1, hidden_channels),
            ('vendors','supply','products'):SAGEConv(-1, hidden_channels)
           # ('vendors','selfloop','vendors'):SAGEConv(-1, hidden_channels)          
            }, aggr='mean') 
        if False:
            self.convs = HeteroConv({
                ('subcats','childof','cats'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1),
                ('depts','childof','subcats'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1),
                ('subdepts','childof','depts'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1),
                ('classes','childof','subdepts'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1),
                ('products','childof','classes'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1),
                ('vendors','supply','products'):GATConv(-1, hidden_channels,add_self_loops=False,heads=1)
            # ('vendors','selfloop','vendors'):SAGEConv(-1, hidden_channels)          
                }, aggr='mean') 
       
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x_dict, edge_index_dict):
        # Pass data through HeteroConv
        x_dict = self.convs(x_dict, edge_index_dict)
        # Apply a simple transformation (e.g., a linear layer)
        out = {key: self.linear(x) for key, x in x_dict.items()}
        return out

    def forward2(self, x_dict, edge_index_dict):
        # Apply heterogeneous convolution
        x_dict = self.convs(x_dict, edge_index_dict)
        # Apply a linear layer to each node type
        for node_type in x_dict:
            x_dict[node_type] = self.lin(x_dict[node_type])
        return x_dict



class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,heads=1):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        #self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = F.relu(x)
        #x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=cfg.gnn.dropout)
        #self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=cfg.gnn.dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = F.elu(x)
        #x = self.conv2(x, edge_index)
        return x
class MHParentPredictor(torch.nn.Module):
    def __init__(self, cfg,pdt_input_dim,mhp_input_dim,hidden_dim,num_mhparents):
        super(MHParentPredictor, self).__init__()
        self.pdt_tfrm=nn.Linear(pdt_input_dim,hidden_dim)
        self.relu=nn.ReLU()
        self.mhp_tfrm=nn.Linear(mhp_input_dim,hidden_dim)
        self.pdt_mhp=nn.Linear(hidden_dim,num_mhparents)
        #self.sm=nn.Softmax(dim=1)

        self.attention = nn.Linear(pdt_input_dim + mhp_input_dim, 1)
        self.product_transform = nn.Linear(pdt_input_dim, hidden_dim)
        self.combined_transform = nn.Linear(hidden_dim + mhp_input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_mhparents)
        
    def forward_cossim(self,pdt,mhp,edge_index=None):
        
        
        pdt=self.pdt_tfrm(pdt)
        pdt = F.normalize(pdt, p=2, dim=1)  # L2 normalize
        #p=self.relu(p)
        mhp=self.mhp_tfrm(mhp)
        mhp=F.normalize(mhp, p=2, dim=1)  # L2 normalize
        #mhp=self.relu(mhp)
        
        scores = torch.matmul(pdt, mhp.T)
        #probs = F.softmax(scores, dim=1)


        return scores
    def forward(self,product_features,class_tensor,edge_index=None):

        num_products=product_features.shape[0]
        src_nodes=edge_index[0]
        class_features=class_tensor[edge_index[1]]
       
        combined_features = torch.cat([product_features, class_features], dim=1)  # Shape: (num_edges, product_features + class_features)
        attention_scores = self.attention(combined_features)  # Shape: (num_edges, 1)
        attention_weights = F.softmax(attention_scores, dim=0)  # Normalize scores across edges

        # Step 2: Compute weighted class features
        weighted_class_features = attention_weights * class_features  # Shape: (num_edges, class_features)

        # Step 3: Aggregate weighted class features for each product node
        aggregated_parents = torch.zeros(num_products, class_features.size(1), device=product_features.device)  # Shape: (num_products, class_features)
        index_tensor = torch.arange(num_products) 
        aggregated_parents.index_add_(0, index_tensor, weighted_class_features)  # Aggregation by product index

        # Step 4: Transform product features and combine with aggregated parent features
        product_features = self.pdt_tfrm(product_features)  # Shape: (num_edges, hidden_dim)
        combined_features = torch.cat([product_features, aggregated_parents[index_tensor]], dim=1)  # Shape: (num_edges, hidden_dim + class_features)
        combined_features = self.combined_transform(combined_features)  # Shape: (num_edges, hidden_dim)

        # Step 5: Produce logits for each product node
        logits = self.output_layer(combined_features)  # Shape: (num_products, output_dim)
        return logits


class MHCosSimParentPredictor(nn.Module): #use cosine similarity insted of mhparentpredictor
    def __init__(self, product_embeddings, class_embeddings):
        super(MHCosSimParentPredictor, self).__init__()
        #self.input_to_embedding = nn.Linear(input_dim, class_embeddings.shape[1])  # Input to match class embedding dim
        self.input_to_embedding = nn.Parameter(product_embeddings, requires_grad=True)
        self.class_embeddings = nn.Parameter(class_embeddings, requires_grad=True)  # Fixed class embeddings

    def forward(self, x):
        # Compute input embedding
        input_embedding = self.input_to_embedding(x)  # Shape: (batch_size, embedding_dim)
        input_embedding = F.normalize(input_embedding, p=2, dim=1)  # L2 normalize
        
        # Normalize class embeddings
        class_embeddings = F.normalize(self.class_embeddings, p=2, dim=1)  # Shape: (num_classes, embedding_dim)
        
        # Compute cosine similarity
        similarities = torch.matmul(input_embedding, class_embeddings.T)  # Shape: (batch_size, num_classes)
        return similarities

    
class GNN_MHP(torch.nn.Module):
    def __init__(self, cfg,gnnmodel,mhpmodel):
        super(GNN_MHP, self).__init__()
        self.gnnmodel=gnnmodel
        self.mhpmodel=mhpmodel
        #self.input_to_embedding = nn.Linear(384, 1393)
        
        self.model_type=gnnmodel.model_type #HETRO,GAT,GRAPHSAGE
 
    #def forward(self,pdt,mhp):
    def forward(self, data,edge_index=None,parenttype='classes'):

        node_embed=self.gnnmodel(data,edge_index)
        if self.model_type=="MH_HETRO":
            pdt_embed=node_embed['products'][edge_index[0]]  #just get the data that exists in edge_index in that sequence
            #class_idx=edge_index[1].unique()
            #class_embed=node_embed['classes']  #class Labels
            #class_embed=node_embed['products'][edge_index[1]] 
            class_embed=node_embed[parenttype]  #class Labels
            
        else:
            pass
        #out=MHCosSimParentPredictor
       #mhpmodel=MHCosSimParentPredictor(pdt_embed,class_embed)
        #out=mhpmodel(pdt_embed)
        if False:
            #pdt_embed=self.input_to_embedding(pdt_embed)
            pdt_embed = F.normalize(pdt_embed, p=2, dim=1)  # L2 normalize
            class_embed = F.normalize(class_embed, p=2, dim=1)  # Shape: (num_classes, embedding_dim)
  
            out = torch.matmul(pdt_embed, class_embed.T)  # Shape: (batch_size, num_classes)

        if True:
        #out=self.mhpmodel(pdt_embed)
            out=self.mhpmodel(pdt_embed,class_embed,edge_index=edge_index)
        return out

    
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cfg,emb=False):
        super(GNNStack, self).__init__()
        model_type=cfg.model.type
        self.model_type=model_type
        num_layers=cfg.gnn.layers_mp
        heads=cfg.gnn.att_heads
        do=cfg.gnn.dropout

        conv_model = self.build_conv_model(model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim,heads=heads))
        assert (num_layers >= 1), 'Number of layers is not >=1'
        for l in range(num_layers-1):
            self.convs.append(conv_model(heads * hidden_dim, hidden_dim))

            

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(heads * hidden_dim, hidden_dim).float(), nn.Dropout(do),
            nn.Linear(hidden_dim, output_dim).float())
        self.relu=nn.LeakyReLU()
        self.dropout = do
        self.num_layers = num_layers
        self.bn=nn.BatchNorm1d(hidden_dim,eps=1)

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            print("using GraphSage")
            return GraphSAGE
        elif model_type == 'GAT':
            print("using GAT")
            return GAT
            # When applying GAT with num heads > 1, you need to modify the
            # input and output dimension of the conv layers (self.convs),
            # to ensure that the input dim of the next layer is num heads
            # multiplied by the output dim of the previous layer.
            # HINT: In case you want to play with multiheads, you need to change the for-loop that builds up self.convs to be
            # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)),
            # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
            pass
            # return GAT
        elif model_type == 'HETRO':
            print("using HETRO GNN")
            return HeteroGNN
        elif model_type=='MH_HETRO':
            print("using MH HETRO GNN")
            return HeteroGNN

    
    def forward(self, data,edge_index_to_use=None):
        if self.model_type=="HETRO" or self.model_type=="MH_HETRO":
            #out=self.forward_hgnn(self,x,edge_index_to_use)
            #return out
            x_dict=data.x_dict
            for i in range(self.num_layers):
                x_dict = self.convs[i](data.x_dict, data.edge_index_dict)
                x_dict = {key: self.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
                for key,x in x_dict.items():
                    if key=='cats':
                        pass  #skip bn for  categories 
                    else:
                        #x_dict[key]=self.bn(x)
                        x_dict[key]=x  #commented bn for now
                        pass
                
            x_dict = {key: self.post_mp(x) for key, x in x_dict.items()}
    

            if self.emb == True:
                return x_dict
            
            x_dict_sm = {key: F.log_softmax(x, dim=1) for key, x in x_dict.items()}
            return x_dict_sm
        else:
            edge_index=edge_index_to_use
            x=data
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout,training=self.training)
            x = self.post_mp(x)

            if self.emb == True:
                return x

            return F.log_softmax(x, dim=1)
            #out=self.forward_gnn(self,x)
            #return out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    def edge_score(self, edge_index, src_embeddings, dst_embeddings):
        """
        Compute scores for edges based on source and destination node embeddings.

        Args:
            edge_index (torch.Tensor): The edge indices (2 x num_edges).
            src_embeddings (torch.Tensor): Embeddings for the source nodes.
            dst_embeddings (torch.Tensor): Embeddings for the destination nodes.

        Returns:
            torch.Tensor: Scores for each edge.
        """
        # Extract source and destination node embeddings for given edge indices
        src = src_embeddings[edge_index[0]]  # Source node embeddings
        dst = dst_embeddings[edge_index[1]]  # Destination node embeddings

        #pos_scores = (out[pos_edge_index[0]] * out[pos_edge_index[1]]).sum(dim=1)
        #neg_scores = (out[neg_edge_index[0]] * out[neg_edge_index[1]]).sum(dim=1)

        # Compute edge scores (dot product in this example)
        scores = (src * dst).sum(dim=1)  # Element-wise multiplication and summation
        return scores
