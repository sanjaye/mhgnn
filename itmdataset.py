import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import os
import networkx as nx
from deepsnap.graph import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
import json
class ItemDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,split=None,openai=True):

        self.node_file = os.path.join(root, 'all_nodes_202411072100.csv')
        self.edge_file = os.path.join(root, 'd_all_edges_202411072100.csv')

        self.rootdir=root
        self.openai=openai
        if self.openai:
            self.save_flle="openai_data.pt"
        else:
            self.save_flle="data.pt"

        #tr,tst,val=split[0],split[1],split[2] #0.75,0.15,0.15  # default values
        if split is not None:
            tr,tst,val=split[0],split[1],split[2] 
        else:
            tr,tst,val=0.70,0.15,0.15
            
        self.train_split,self.test_split,self.val_split=tr,tst,val
        
        super().__init__(root, transform, pre_transform)
        self.data, self.G = torch.load(self.processed_paths[0],weights_only=False)



    @property
    def raw_file_names(self):
        return ['node1.csv', 'edge1.csv']

    @property
    def processed_file_names(self):
        return [self.save_flle]

    def download(self):
        # Implement this if data needs to be downloaded
        pass

    def process(self):
        # Load node and edge data
        nodes_df = pd.read_csv(self.node_file)
        edges_df = pd.read_csv(self.edge_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.openai:
            openai_embed_file="text_embeddings.json"
            embed_file=os.path.join(self.rootdir, openai_embed_file)
            # Load the JSON file with embeddings
            with open(embed_file, 'r') as f:
                embeddings_json = json.load(f)

            # Convert JSON values to tensors if needed
            for key in embeddings_json:
                embeddings_json[key] = torch.tensor(embeddings_json[key])

            embedding_size = len(next(iter(embeddings_json.values())))  

            def get_openai_first_or_random(nodename):
                #totalcnt+=1
                try:
                    return embeddings_json[str(nodename)]
                except KeyError:
                    #randomcnt+=1
                    return torch.rand(embedding_size) 

            nodes_df["emb1"]=nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))      
        else:
                     # Process node features
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.to(device)
            #desc=nodes_df["nodename"]
            #nodes_df["emb1"]=nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
        
            nodes_df["emb1"]=nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
    
        #nodes_df["emb1"] = nodes_df['nodeid'].apply(lambda _: torch.tensor(np.random.rand(384),dtype=torch.float))
        feature_dim=nodes_df["emb1"][0].shape[0]  
        x = torch.stack(nodes_df['emb1'].values.tolist())
        #x = torch.tensor(nodes_df['features'].apply(eval).tolist(), dtype=torch.float)
        num_nodes = x.size(0)
        num_edges=len(edges_df)

        # Process edge connections
        edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
        #train_mask val_mask,test_mask
        # Create a PyTorch Geometric Data object
      
        edge_attr=torch.tensor([0,1,2,3,4,5,6],dtype=torch.long)  # Relationship types for the edges 0-cat-sub 1-sub-dept 2 dept-subdept 3 subdept-cl 4-cl-sc 5 sc-item 6-vendor item etc)
 
 
        indices = torch.randperm(num_nodes)  # Randomly shuffle the indices
        edge_indices= torch.randperm(num_edges)

        # Split into 70% train, 15% validation, 15% test
        train_size,edge_train_size = int(self.train_split * num_nodes),int(self.train_split * num_edges)
        val_size,edge_val_size = int(self.val_split * num_nodes),int(self.val_split  * num_edges)
        #test_size,edge_test_size = num_nodes - train_size - val_size, num_edges-edge_train_size-edge_val_size



        train_indices,edge_train_indices = indices[:train_size], edge_indices[:edge_train_size]
        val_indices,edge_val_indices = indices[train_size:train_size + val_size], edge_indices[edge_train_size:edge_train_size + edge_val_size]
        test_indices,edge_test_indices = indices[train_size + val_size:],edge_indices[edge_train_size + edge_val_size:]

        # Create masks
        train_mask,edge_train_mask = torch.zeros(num_nodes, dtype=torch.bool),torch.zeros(num_edges, dtype=torch.bool)
        val_mask,edge_val_mask = torch.zeros(num_nodes, dtype=torch.bool),torch.zeros(num_edges, dtype=torch.bool)
        test_mask,edge_test_mask = torch.zeros(num_nodes, dtype=torch.bool),torch.zeros(num_edges, dtype=torch.bool)

        train_mask[train_indices], edge_train_mask[edge_train_indices]= True,True
        val_mask[val_indices],edge_val_mask[edge_val_indices] = True,True
        test_mask[test_indices],edge_test_mask[edge_test_indices] = True,True



        feature_dim=10
        node_labels=torch.tensor(nodes_df["lvl"])
        edge_type_labels=torch.tensor(edges_df["edgetype"])

        data = Data(x=x, edge_index=edge_index,y=node_labels,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,edge_attr=edge_attr,
                    edge_type_labels=edge_type_labels,edge_train_mask=edge_train_mask,edge_val_mask=edge_val_mask,edge_test_mask=edge_test_mask)
        
        data.num_nodes = num_nodes
        data.num_classes=7
        data.num_node_features=feature_dim
        data.num_edge_classes=edge_attr.shape[0]  #6

      
       # c = torch.randint(0, 5, (num_nodes,))
        #node_labels=torch.tensor(nodes_df["lvl"])

        # Convert to networkx graph with DeepSNAP compatibility
        G = nx.Graph()
        default_tensor=torch.zeros(feature_dim) 
        
        for idx, row in nodes_df.iterrows():
            G.add_node(idx, node_feature=x[idx], label=node_labels[idx] ,tensor_attr=default_tensor.clone())
            # row['Node']
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'])
        
        # Wrap the networkx graph in DeepSNAP’s Graph class
        self.G = Graph(G)
        
        #self.G.num_node_features = feature_dim
        self.G.node_label=node_labels

        # Save the Data object and DeepSNAP Graph
        torch.save([data, self.G], self.processed_paths[0])

    def len(self):
        return 1  # Adjust if handling multiple graphs

    def number_of_nodes(self):
        return self.data.num_nodes  # Adjust if handling multiple graphs
    
    def number_of_edges(self):
        return len(self.data.edge_index)  # Adjust if handling multiple graphs
    def nodes(self):
        return self.G._get_node_attributes
    def get(self, idx):
        data, G = torch.load(self.processed_paths[0])
        return data, G
