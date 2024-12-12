import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import os
import networkx as nx
from deepsnap.graph import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops  #required for vendor nodes
import json


class hItemDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,split=None,openai=True):
        self.rootdir=root
        self.openai=openai
        if self.openai:
            self.save_flle="openai_hetero_data.pt"
        else:
            self.save_flle="hetero_data.pt"
        self.node_file = os.path.join(root, 'all_nodes_202411072100.csv')
        self.edge_file = os.path.join(root, 'd_all_edges_202411072100.csv')

        #tr,tst,val=split[0],split[1],split[2] #0.75,0.15,0.15  # default values
        if split is not None:
            tr,tst,val=split[0],split[1],split[2] 
        else:
            tr,tst,val=0.70,0.15,0.15
            
        self.train_split,self.test_split,self.val_split=tr,tst,val
        
        super().__init__(root, transform, pre_transform)

        self.data= torch.load(self.processed_paths[0],weights_only=False)



    @property
    def raw_file_names(self):
        return ['node1.csv', 'edge1.csv']

    @property
    def processed_file_names(self):
        return [self.save_flle]

    def download(self):
        # Implement this if data needs to be downloaded
        pass
    def convert_hetero_to_nx(hetero_data):
        #G = nx.DiGraph()  # Use DiGraph for directed edges, Graph for undirected
        G=nx.Graph()

        # Add nodes
        for node_type in hetero_data.node_types:
            for i, node_features in enumerate(hetero_data[node_type].x):
                G.add_node((node_type, i), **{'features': node_features.tolist()})

        # Add edges
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index
            for src, dst in edge_index.t().tolist():
                G.add_edge((src_type, src), (dst_type, dst), relation=rel_type)

        return G

    def process(self):
        # Load node and edge data
        nodes_df = pd.read_csv(self.node_file)
        edges_df = pd.read_csv(self.edge_file)



        #desc=nodes_df["nodename"]
        
        #nodes_df["emb1"]=nodes_df['Node'].apply(lambda desc: model.encode(str(desc),convert_to_tensor=True))

        vdr_nodes_df=nodes_df[nodes_df['lvl']==6].copy()
        pdt_nodes_df=nodes_df[nodes_df['lvl']==5].copy()
        class_nodes_df=nodes_df[nodes_df['lvl']==4].copy()
        subdept_nodes_df=nodes_df[nodes_df['lvl']==3].copy()
        dept_nodes_df=nodes_df[nodes_df['lvl']==2].copy()
        subcat_nodes_df=nodes_df[nodes_df['lvl']==1].copy()
        cat_nodes_df=nodes_df[nodes_df['lvl']==0].copy()

        offset_cat=max(cat_nodes_df["nodeid"])+1
        offset_subcat=max(subcat_nodes_df["nodeid"])+1
        offset_dept=max(dept_nodes_df["nodeid"])+1
        offset_subdept=max(subdept_nodes_df["nodeid"])+1
        offset_class=max(class_nodes_df["nodeid"])+1
        offset_pdt=max(pdt_nodes_df["nodeid"])+1

        subcat_nodes_df['nodeid'] = subcat_nodes_df['nodeid'] - offset_cat
        dept_nodes_df['nodeid'] = dept_nodes_df['nodeid'] - offset_subcat
        subdept_nodes_df['nodeid'] = subdept_nodes_df['nodeid'] - offset_dept
        class_nodes_df['nodeid'] = class_nodes_df['nodeid'] - offset_subdept
        pdt_nodes_df['nodeid'] = pdt_nodes_df['nodeid'] - offset_class
        vdr_nodes_df['nodeid'] = vdr_nodes_df['nodeid'] - offset_pdt

       




 
        if  self.openai==True:
            openai_embed_file="text_embeddings.json"
            embed_file=os.path.join(self.rootdir, openai_embed_file)
            # Load the JSON file with embeddings
            with open(embed_file, 'r') as f:
                embeddings_json = json.load(f)

            # Convert JSON values to tensors if needed
            for key in embeddings_json:
                embeddings_json[key] = torch.tensor(embeddings_json[key])

            embedding_size = len(next(iter(embeddings_json.values())))
            totalcnt=0
            randomcnt=0
            def get_openai_first_or_random(nodename):
                #totalcnt+=1
                try:
                    return embeddings_json[str(nodename)]
                except KeyError:
                    #randomcnt+=1
                    return torch.rand(embedding_size) 

            pdt_nodes_df["emb1"]=pdt_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            feature_dim=embedding_size
            pdt_x = torch.stack(pdt_nodes_df['emb1'].values.tolist())
            num_pdt_nodes = pdt_x.size(0)

            vdr_nodes_df["emb1"]=vdr_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            #feature_dim=vdr_nodes_df["emb1"][0].shape[0]  
            vdr_x = torch.stack(vdr_nodes_df['emb1'].values.tolist())
            num_vdr_nodes = vdr_x.size(0)



            class_nodes_df["emb1"]=class_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            class_x = torch.stack(class_nodes_df['emb1'].values.tolist())
            num_class_nodes=class_x.size(0)

            subdept_nodes_df["emb1"]=subdept_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            subdept_x = torch.stack(subdept_nodes_df['emb1'].values.tolist())
            num_subdept_nodes=subdept_x.size(0)

            dept_nodes_df["emb1"]=dept_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            dept_x = torch.stack(dept_nodes_df['emb1'].values.tolist())
            num_dept_nodes=dept_x.size(0)

            subcat_nodes_df["emb1"]=subcat_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            subcat_x = torch.stack(subcat_nodes_df['emb1'].values.tolist())
            num_subdept_nodes=subcat_x.size(0)

            cat_nodes_df["emb1"]=cat_nodes_df['nodename'].apply(lambda nodename: get_openai_first_or_random(nodename))
            cat_x = torch.stack(cat_nodes_df['emb1'].values.tolist())
            num_cat_nodes=cat_x.size(0)

            print("total openai embedcount,randomembedcount",totalcnt,randomcnt)

        else:#use sentence transformer mebdding
                    
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Process node features
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model=model.to(device)

            pdt_nodes_df["emb1"]=pdt_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            #feature_dim=pdt_nodes_df["emb1"][0].shape[0]  
            pdt_x = torch.stack(pdt_nodes_df['emb1'].values.tolist())
            num_pdt_nodes = pdt_x.size(0)

            vdr_nodes_df["emb1"]=vdr_nodes_df['nodeid'].apply(lambda _: torch.tensor(np.random.rand(feature_dim),dtype=torch.float)) #random for vendor
            #feature_dim=vdr_nodes_df["emb1"][0].shape[0]  
            vdr_x = torch.stack(vdr_nodes_df['emb1'].values.tolist())
            num_vdr_nodes = vdr_x.size(0)



            class_nodes_df["emb1"]=class_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            class_x = torch.stack(class_nodes_df['emb1'].values.tolist())
            num_class_nodes=class_x.size(0)

            subdept_nodes_df["emb1"]=subdept_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            subdept_x = torch.stack(subdept_nodes_df['emb1'].values.tolist())
            num_subdept_nodes=subdept_x.size(0)

            dept_nodes_df["emb1"]=dept_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            dept_x = torch.stack(dept_nodes_df['emb1'].values.tolist())
            num_dept_nodes=dept_x.size(0)

            subcat_nodes_df["emb1"]=subcat_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            subcat_x = torch.stack(subcat_nodes_df['emb1'].values.tolist())
            num_subdept_nodes=subcat_x.size(0)

            cat_nodes_df["emb1"]=cat_nodes_df['nodename'].apply(lambda nodename: model.encode(str(nodename),convert_to_tensor=True))
            cat_x = torch.stack(cat_nodes_df['emb1'].values.tolist())
            num_cat_nodes=cat_x.size(0)

        pdt_indices = torch.randperm(num_pdt_nodes)  # Randomly shuffle the indices
        train_pdt_size=int(self.train_split * num_pdt_nodes)
        val_pdt_size=int(self.val_split * num_pdt_nodes)
        
        train_pdt_indices = pdt_indices[:train_pdt_size]
        val_pdt_indices  = pdt_indices[train_pdt_size:train_pdt_size + val_pdt_size] 
        test_pdt_indices = pdt_indices[train_pdt_size + val_pdt_size:] 

        # Create masks
        train_pdt_mask = torch.zeros(num_pdt_nodes, dtype=torch.bool) 
        val_pdt_mask  = torch.zeros(num_pdt_nodes, dtype=torch.bool) 
        test_pdt_mask  = torch.zeros(num_pdt_nodes, dtype=torch.bool)

        train_pdt_mask[train_pdt_indices] = True
        val_pdt_mask[val_pdt_indices]  = True
        test_pdt_mask[test_pdt_indices] = True

        hdata=HeteroData()
        hdata['products'].x=pdt_x
        hdata['products'].train_mask=train_pdt_mask
        hdata['products'].val_mask=val_pdt_mask
        hdata['products'].test_mask=test_pdt_mask
        
        hdata['vendors'].x=vdr_x
        hdata['classes'].x=class_x
        hdata['subdepts'].x=subdept_x
        hdata['depts'].x=dept_x
        hdata['subcats'].x=subcat_x
        hdata['cats'].x=cat_x

        edge_cat_subcat_df=edges_df[edges_df['edgetype']==0]
        edge_subcat_dept_df=edges_df[edges_df['edgetype']==1]
        edge_dept_subdept_df=edges_df[edges_df['edgetype']==2]
        edge_subdept_class_df=edges_df[edges_df['edgetype']==3]
        edge_class_pdt_df=edges_df[edges_df['edgetype']==4]
        edge_pdt_vdr_df=edges_df[edges_df['edgetype']==5]

        #edge_subcat_cat_df['source'] = edge_subcat_cat_df['source'] - offset_cat
        edge_cat_subcat_df['target'] = edge_cat_subcat_df['target'] - offset_cat #subcat

        edge_subcat_dept_df['source'] = edge_subcat_dept_df['source'] - offset_cat #subcat
        edge_subcat_dept_df['target'] = edge_subcat_dept_df['target'] - offset_subcat #dept

        edge_dept_subdept_df['source'] = edge_dept_subdept_df['source'] - offset_subcat #dept
        edge_dept_subdept_df['target'] = edge_dept_subdept_df['target'] - offset_dept #subdept

        edge_subdept_class_df['source'] = edge_subdept_class_df['source'] - offset_dept #subdept
        edge_subdept_class_df['target'] = edge_subdept_class_df['target'] - offset_subdept #class
        
        edge_class_pdt_df['source'] = edge_class_pdt_df['source'] - offset_subdept #class
        edge_class_pdt_df['target'] = edge_class_pdt_df['target'] - offset_class #pdt

        edge_pdt_vdr_df['source'] = edge_pdt_vdr_df['source'] - offset_class #class
        edge_pdt_vdr_df['target'] = edge_pdt_vdr_df['target'] - offset_pdt #pdt


        edge_index_cat_subcat=torch.tensor(edge_cat_subcat_df[['target', 'source']].values.T, dtype=torch.long)
        edge_index_subcat_dept=torch.tensor(edge_subcat_dept_df[['target', 'source']].values.T, dtype=torch.long)
        edge_index_dept_subdept=torch.tensor(edge_dept_subdept_df[['target', 'source']].values.T, dtype=torch.long)
        edge_index_subdept_class=torch.tensor(edge_subdept_class_df[['target', 'source']].values.T, dtype=torch.long)
        edge_index_class_pdt=torch.tensor(edge_class_pdt_df[['target', 'source']].values.T, dtype=torch.long)
        edge_index_pdt_vdr=torch.tensor(edge_pdt_vdr_df[['target', 'source']].values.T, dtype=torch.long)
       
       
 

        if True:
            print("node counts, cat,subcat,dept,subdept,class,pdt,vdr",len(cat_nodes_df),len(subcat_nodes_df),
                  len(dept_nodes_df),len(subdept_nodes_df),len(class_nodes_df),len(pdt_nodes_df),len(vdr_nodes_df))
            print("offsets cat,subcat,dept,subdept,class,pdt",offset_cat,offset_subcat,offset_dept,offset_subdept,offset_class,offset_pdt)
            print("edges cs,scd,dsd,sdc,cpdt,pdtvdr",edge_index_cat_subcat.shape,edge_index_subcat_dept.shape,edge_index_dept_subdept.shape,
                  edge_index_subdept_class.shape,edge_index_class_pdt.shape,edge_index_pdt_vdr.shape)
        
        hdata['subcats','childof','cats'].edge_index=edge_index_cat_subcat
        hdata['depts','childof','subcats'].edge_index=edge_index_subcat_dept
        hdata['subdepts','childof','depts'].edge_index=edge_index_dept_subdept
        hdata['classes','childof','subdepts'].edge_index=edge_index_subdept_class
        hdata['products','childof','classes'].edge_index=edge_index_class_pdt
        hdata['vendors','supply','products'].edge_index=edge_index_pdt_vdr

        # Iterate over all edge types and add self-loops
   


       
        #vendor_sl_index, _ = add_self_loops(hdata.vendors.edge_index)
        #h#data['vendors','selfloop','vendors'].edge_index=vendor_sl_index
        
        
        num_edges=edge_index_class_pdt.shape[1]  # in edge_index dim=1 has the # of edges
        edge_indices= torch.randperm(num_edges)
        train_edge_size,val_edge_size,test_edge_size=int(self.train_split*num_edges),int(self.val_split*num_edges),int(self.test_split*num_edges)
        train_edge_indices=edge_indices[:train_edge_size]
        val_edge_indices=edge_indices[train_edge_size:train_edge_size + val_edge_size]
        test_edge_indices= edge_indices[train_edge_size + val_edge_size:]
     
        train_pdt_edge_mask = torch.zeros(num_edges, dtype=torch.bool) 
        val_pdt_edge_mask  = torch.zeros(num_edges, dtype=torch.bool) 
        test_pdt_edge_mask  = torch.zeros(num_edges, dtype=torch.bool)

        train_pdt_mask[train_edge_indices] = True
        val_pdt_mask[val_edge_indices]  = True
        test_pdt_mask[test_edge_indices] = True

        hdata['products','childof','classes'].train_mask=train_pdt_mask
        hdata['products','childof','classes'].val_mask=val_pdt_mask
        hdata['products','childof','classes'].test_mask=test_pdt_mask

        torch.save(hdata, self.save_flle)
        torch.save(hdata,self.processed_paths[0])

        if False:
            # Convert to networkx graph with DeepSNAP compatibility
            G = self.convert_hetero_to_nx(hdata)          
            
            # Wrap the networkx graph in DeepSNAP’s Graph class
            self.G = Graph(G)
            
            #self.G.num_node_features = feature_dim
            #self.G.node_label=node_labels

            # Save the Data object and DeepSNAP Graph
            torch.save([hdata, self.G], self.processed_paths[0])


    def len(self):
        return 1  # Adjust if handling multiple graphs

    def number_of_nodes(self):
        return self.data.num_nodes  # Adjust if handling multiple graphs
    
    def number_of_edges(self):
        return len(self.data.edge_index)  # Adjust if handling multiple graphs
    def nodes(self):
        return self.G._get_node_attributes
    def get(self, idx):
        data= torch.load(self.processed_paths[0])
        return data
