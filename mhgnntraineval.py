import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
#from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.utils import negative_sampling
from hetroitmdataset import hItemDataset
from itmdataset import ItemDataset
from tqdm import trange
import copy
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn

from cmd_args import parse_args
from config import cfg,load_cfg,set_out_dir,dump_cfg

from mhgnns import GNNStack,MHParentPredictor,MHGNN


from torch_geometric.nn import HeteroConv, GCNConv

#from torchviz import make_dot


def get_negative_edges(edge_index, num_nodes, num_neg_samples):
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples
    )
    return neg_edge_index

def compute_loss(pos_score, neg_score):
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))  #works better than logsoftmax
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss

def custom_negative_sampling(pos_edges, num_nodes_src, num_nodes_tgt, num_neg_samples):
    # Set to track positive edges
    pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    neg_edges = []
    
    while len(neg_edges) < num_neg_samples:
        # Randomly sample source and target nodes
        src = torch.randint(0, num_nodes_src, (1,)).item()
        tgt = torch.randint(0, num_nodes_tgt, (1,)).item()
        if (src, tgt) not in pos_set:  # Ensure it's not a positive edge
            neg_edges.append((src, tgt))
            pos_set.add((src, tgt))  # Avoid duplicates
    
    # Convert to tensor
    neg_edges = torch.tensor(neg_edges).t()
    return neg_edges

def train_hgnn(train_data,val_data,device,cfg):

    

    num_node_features=train_data.x_dict['products'].shape[1] #384

    #node_embedding = nn.Parameter(product_embeddings, requires_grad=True)
    
    model=GNNStack(input_dim=num_node_features,hidden_dim=cfg.gnn.dim_inner,output_dim=num_node_features,cfg=cfg,emb=True)
        
    model=model.to(device)


   # data.data.to(device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)

    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    train_data=train_data.to(device)
    for epoch in trange(cfg.optim.max_epoch, desc="Training", unit="Epochs"):
        model.train()
        total_loss = 0

        optimizer.zero_grad()

        out_dict = model(train_data)
     

        for edge_type, edge_index in train_data.edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type

            if src_type=="vendors" or src_type=="subcats":  #vendors do not have inbound links, cat-subcat link does not provide negatives
                pass
            else:
            # Positive edges
                edgeloss=0

                pos_neg_edges = train_data[edge_type].edge_label_index
                pos_neg_score = model.edge_score(pos_neg_edges, out_dict[src_type], out_dict[dst_type])
                #edgeloss = compute_loss(pos_neg_score, train_data[edge_type].edge_label)
                edgeloss = F.binary_cross_entropy_with_logits(pos_neg_score, train_data[edge_type].edge_label)  #works better than logsoftmax
                total_loss += edgeloss  # Accumulate the loss

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss)
        if epoch % 10 == 0:
          #test_acc = test(loader=test_loader, test_model=model,ds=dataset)
          #print("epoch,train_loss",epoch,loss)
          f1,prec,recall = evaluate_hgnn(model,val_data,device)
          test_accs.append(f1)
          print("epoch,train_loss",epoch,total_loss.cpu().detach().numpy())
          print("test scores: epoch, f1,prec,recall",epoch,total_loss.cpu().detach().numpy(),round(f1,2),prec,recall)
          
          if f1 > best_acc:
            best_acc = f1
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc
    #return loss.item()

# Evaluation loop
def evaluate_hgnn(model,data, device,is_validation=False, save_model_preds=False,config_filename="default"):
    #model=test_model
    
    model.eval()
    data=data.to(device)
    with torch.no_grad():
        out_dict = model(data) 
        true_edge_score_dict={}
        pred_edge_score_dict={}
        edge_loss_dict={}
        data_l_dict={}
        total_loss=0
        
        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type

            if src_type=="vendors" or src_type=="subcats":  #vendors do not have inbound links, cat-subcat link does not provide negatives
                pass
            else:
                # Positive edges
                pos_neg_edges = data[edge_type].edge_label_index
                pos_neg_score = model.edge_score(pos_neg_edges, out_dict[src_type], out_dict[dst_type])

                edgeloss = compute_loss(pos_neg_score, data[edge_type].edge_label)
                total_loss=edgeloss
                edge_loss_dict[src_type]=edgeloss

                y_pred=(pos_neg_score>0).float()
                y_true=data[edge_type].edge_label

                true_edge_score_dict[src_type]=y_true
                pred_edge_score_dict[src_type]=y_pred

                data_l=data[edge_type].edge_label_index
                data_l=torch.cat([data_l,y_true.unsqueeze(dim=0)])
                data_l=torch.cat([data_l,y_pred.unsqueeze(dim=0)])

                data_l_dict[src_type]=data_l

            
        
        combined_y_true=torch.cat(list(true_edge_score_dict.values()), dim=0)
        combined_y_pred=torch.cat(list(pred_edge_score_dict.values()), dim=0)
        combined_data_l=torch.cat(list(data_l_dict.values()), dim=1)
        combined_edge_loss=list(edge_loss_dict.values())
        
        precision = precision_score(combined_y_true.cpu(), combined_y_pred.cpu())
        recall = recall_score(combined_y_true.cpu(), combined_y_pred.cpu())
        f1 = f1_score(combined_y_true.cpu(), combined_y_pred.cpu())
        if save_model_preds:
            print ("Saving Model Predictions for config filename", config_filename)

            data = {}
            data['pred'] = combined_y_pred.view(-1).cpu().detach().numpy()
            data['label'] = combined_y_true.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=combined_data_l.cpu())
            # Save locally as csv 
            current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
            df.to_csv('results\itmmh-link-' + config_filename +"-" + current_datetime + '.csv', sep=',', index=False)

#mask = data.val_mask if is_validation else data.test_mask

        return f1,precision,recall
        

        # Convert scores to binary predictions





        # Compute F1 score



def train_evaluate_hgnn(config_filename,config_obj,device):

    cfg=config_obj

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device="cpu"

    ds_data = hItemDataset(root=cfg.dataset.dir,split=cfg.dataset.split,openai=cfg.dataset.open_ai_embed)

    #ds= [tup[0] for tup in ds_data_graph]

    #mhdata=ds[0]

    # Create a PyTorch Geometric Data object
    #data = Data(x=x, edge_index=edge_index)
    #data=Data(ds_data.data.x,ds_data.data.edge_index)

    #data = train_test_split_edges(data=data,val_ratio=cfg.dataset.valratio)  # Split the edges into train, val, and test sets

    # Train the model

    transform = RandomLinkSplit(
        num_val=0.15,                  # Proportion of edges for validation
        num_test=0.15,                 # Proportion of edges for testing
        is_undirected=False,          # Adjust based on your graph's nature
        add_negative_train_samples=True,  # Add negative samples for training
        edge_types=[('subcats', 'childof', 'cats'),('depts', 'childof', 'subcats'),
                    ('subdepts', 'childof', 'depts'),('classes', 'childof', 'subdepts'),
                    ('products', 'childof', 'classes'),('vendors', 'supply', 'products')]
    )

    train_data, val_data, test_data = transform(ds_data.data)
    test_accs, losses, best_model, best_acc = train_hgnn(train_data=train_data,val_data=val_data,device=device,cfg=cfg)

    #use the best model to save the scores and plot

    f1_score,prec,recall=evaluate_hgnn(best_model,data=test_data,device=device,save_model_preds=True,config_filename=config_filename)
    #, ds_data,device,is_validation=False, save_model_preds=True,config_filename=config_filename)

    plt.title("MH Node link Prediction for " + config_filename)
    losses = [tensor.item() for tensor in losses]
    #losses.to(device)
    #test_accs.to(device)dg
    plt.plot(losses, label="training loss" + " - " + cfg.model.type)
    plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('results\itmmh-Node-'+ config_filename + "-" + current_datetime + '.png')
    return f1_score,prec,recall,losses,test_accs


def train_gnn(train_data,val_data,device,cfg):

    
    #test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_node_features=train_data.x.shape[1]
    model=GNNStack(input_dim=num_node_features,hidden_dim=cfg.gnn.dim_inner,output_dim=num_node_features,cfg=cfg,emb=True)
           
    model=model.to(device)
    train_data=train_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)

    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(cfg.optim.max_epoch, desc="Training", unit="Epochs"):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index) 
        pos_neg_score=model.edge_score(train_data.edge_label_index,out,out)  #use both positive and -ve edges
        #loss=model.loss(pos_neg_score,data.edge_label)
        loss = F.binary_cross_entropy_with_logits(pos_neg_score, train_data.edge_label)  #works better than logsoftmax
   
        loss.backward()
        optimizer.step()
        losses.append(loss)
        if epoch % 10 == 0:
          #test_acc = test(loader=test_loader, test_model=model,ds=dataset)
          #print("epoch,train_loss",epoch,loss)
          f1,prec,recall = evaluate_gnn(model=model,data=val_data,device=device)
          test_accs.append(f1)
          print("epoch,train_loss",epoch,loss.cpu().detach().numpy())
          print("test scores: epoch, f1,prec,recall",epoch,loss.cpu().detach().numpy(),round(f1,2),prec,recall)
          
          if f1 > best_acc:
            best_acc = f1
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc
    #return loss.item()

# Evaluation loop
def evaluate_gnn(model,data, device,is_validation=False, save_model_preds=False,config_filename="default"):
    #model=test_model
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        # Compute scores for positive and negative edges

        pos_neg_score=model.edge_score(data.edge_label_index,out,out)  #use both positive and -ve edges


        y_pred=pos_neg_score

        y_pred = (y_pred > 0).float()
        y_true=data.edge_label        

 
        data_l=data.edge_label_index
        data_l=torch.cat([data_l,y_true.unsqueeze(dim=0)])
        data_l=torch.cat([data_l,y_pred.unsqueeze(dim=0)])


        # Compute F1 score
        precision = precision_score(y_true.cpu(), y_pred.cpu())
        recall = recall_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        if save_model_preds:
            print ("Saving Model Predictions for config filename", config_filename)

            data = {}
            data['pred'] = y_pred.view(-1).cpu().detach().numpy()
            data['label'] = y_true.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data_l.cpu())
            # Save locally as csv 
            current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
            df.to_csv('results\itmmh-link-' + config_filename +"-" + current_datetime + '.csv', sep=',', index=False)

        #mask = data.val_mask if is_validation else data.test_mask

        return f1,precision,recall
def train_evaluate_gnn(config_filename,config_obj,device):

    cfg=config_obj

   

    ds_data_graph = ItemDataset(root=cfg.dataset.dir,split=cfg.dataset.split,openai=cfg.dataset.open_ai_embed)

    ds= [tup[0] for tup in ds_data_graph]

    mhdata=ds[0]

    # Create a PyTorch Geometric Data object
    #data = Data(x=x, edge_index=edge_index)
    data=Data(mhdata.x,mhdata.edge_index)

    #data = train_test_split_edges(data=data,val_ratio=cfg.dataset.valratio)  # Split the edges into train, val, and test sets

    transform = RandomLinkSplit(is_undirected=False, num_val=0.15,num_test=0.15, add_negative_train_samples=True)  

    train_data, val_data, test_data = transform(data)

    train_data=train_data.to(device)
    test_data=test_data.to(device)
    val_data=val_data.to(device)

    # Train the model
    test_accs, train_losses, best_model, best_acc = train_gnn(train_data=train_data,val_data=val_data,device=device,cfg=cfg)
 

    #use the best model to save the scores and plot

    f1_score,prec,recall=evaluate_gnn(model=best_model, data=test_data,device=device,is_validation=False, save_model_preds=True,config_filename=config_filename)

    plt.title("MH Node link Prediction for " + config_filename)
    losses = [tensor.item() for tensor in train_losses]
    #losses.to(device)
    #test_accs.to(device)dg
    plt.plot(losses, label="training loss" + " - " + cfg.model.type)
    plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)

    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('results\itmmh-Node-'+ config_filename + "-" + current_datetime + '.png')
    return f1_score,prec,recall,losses,test_accs

def get_product_parent_edge_index(hdata,parenttype):
       
# Get the existing edge indices
    products_to_classes = hdata['products', 'childof', 'classes'].edge_index
    classes_to_subdepts = hdata['classes', 'childof', 'subdepts'].edge_index
    subdepts_to_depts = hdata['subdepts', 'childof', 'depts'].edge_index
    depts_to_subcats = hdata['depts', 'childof', 'subcats'].edge_index
    subcats_to_cats = hdata['subcats', 'childof', 'cats'].edge_index


     
    products_to_classes_map = {src.item(): dst.item() for src, dst in zip(products_to_classes[0], products_to_classes[1])}
    classes_to_subdepts_map = {src.item(): dst.item() for src, dst in zip(classes_to_subdepts[0], classes_to_subdepts[1])}
    subdepts_to_depts_map = {src.item(): dst.item() for src, dst in zip(subdepts_to_depts[0], subdepts_to_depts[1])}
    depts_to_subcats_map={src.item(): dst.item() for src, dst in zip(depts_to_subcats[0], depts_to_subcats[1])}
    subcats_to_cats_map={src.item(): dst.item() for src, dst in zip(subcats_to_cats[0], subcats_to_cats[1])}

    # Derive product -> dept edges
    product_to_dept_edges = []
    product_to_subcat_edges = []
    product_to_cat_edges = []
    product_to_subdept_edges=[]
    for  product,class_, in products_to_classes_map.items():
        if class_ in classes_to_subdepts_map:
            subdept = classes_to_subdepts_map[class_]
            product_to_subdept_edges.append((product,subdept))
            if subdept in subdepts_to_depts_map:
                dept = subdepts_to_depts_map[subdept]
                product_to_dept_edges.append((product, dept))
                if dept in depts_to_subcats_map:
                    subcat=depts_to_subcats_map[dept]
                    product_to_subcat_edges.append((product,subcat))
                    if subcat in subcats_to_cats_map:
                        cat=subcats_to_cats_map[subcat]
                        product_to_cat_edges.append((product,cat))

    # Convert to tensor

    if product_to_cat_edges:
        product_to_cat_edge_index = torch.tensor(product_to_cat_edges).t()
        hdata['products', 'childof', 'cats'].edge_index = product_to_cat_edge_index

    if product_to_subcat_edges:
        product_to_subcat_edge_index = torch.tensor(product_to_subcat_edges).t()
        hdata['products', 'childof', 'subcats'].edge_index = product_to_subcat_edge_index

    if product_to_dept_edges:
        product_to_dept_edge_index = torch.tensor(product_to_dept_edges).t()
        hdata['products', 'childof', 'depts'].edge_index = product_to_dept_edge_index

    if product_to_subdept_edges:
        product_to_subdept_edge_index = torch.tensor(product_to_subdept_edges).t()
        hdata['products', 'childof', 'subdepts'].edge_index = product_to_subdept_edge_index

    product_to_class_edge_index= hdata['products', 'childof', 'classes'].edge_index  #already in the dataset
    
 
    if parenttype=='cats':
        parent_edge_index=product_to_cat_edge_index
    elif parenttype=='subcats':
        parent_edge_index=product_to_subcat_edge_index
    elif parenttype=='depts':
        parent_edge_index=product_to_dept_edge_index
    elif parenttype=='subdepts':
        parent_edge_index=product_to_subdept_edge_index
    else:
        parent_edge_index=product_to_class_edge_index
 
    return parent_edge_index,hdata

def train_hgnn_mhp(train_data,val_data,device,cfg):

    
    #test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    #num_node_features=data.x.shape[1]

    #num_node_features=384
    num_node_features=train_data.x_dict['products'].shape[1] 

    

    edge_label=train_data[('products','childof','classes')].edge_label
    edge_label_index=train_data[('products','childof','classes')].edge_label_index

    #combined=torch.cat((edge_label_index,torch.unsqueeze(edge_label,dim=0)),dim=0)

    #pdt_indexes=combined[0,combined[2,:]==1]
    #class_indexes=combined[1,combined[2,:]==1]
    parenttype=cfg.model.pred_parent

    parent_edge_index,data_addnl_edges=get_product_parent_edge_index(train_data,parenttype=parenttype )
    if cfg.dataset.add_direct_pdt_edges:
        train_data=data_addnl_edges
 
    
    parent_indexes =parent_edge_index[1]

    parent_indexes=parent_indexes.to(device)
    parent_edge_index=parent_edge_index.to(device)
    num_classes=train_data.x_dict[parenttype].shape[0] 
 

    gnnmodel=GNNStack(input_dim=num_node_features,hidden_dim=cfg.gnn.dim_inner,output_dim=num_node_features,cfg=cfg,emb=True)
  
    mhpmodel=MHParentPredictor(cfg,num_node_features,num_node_features,cfg.gnn.dim_inner,num_classes)
    

    gnn_mhp_model=MHGNN(cfg,gnnmodel,mhpmodel)

    gnnmodel=gnnmodel.to(device)
    mhpmodel=mhpmodel.to(device)
    gnn_mhp_model=gnn_mhp_model.to(device)
    

    optimizer = torch.optim.Adam(gnn_mhp_model.parameters(), lr=cfg.optim.base_lr)
    criterion = nn.CrossEntropyLoss() 

    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    train_data=train_data.to(device)
 

    #parent_indexes=pdt_dpt_edge_index[1]
    best_epoch=0
    for epoch in trange(cfg.optim.max_epoch, desc="Training", unit="Epochs"):
        gnn_mhp_model.train()
        #total_loss = 0
        optimizer.zero_grad()

        #out = gnn_mhp_model(train_data,edge_label_index)
        out = gnn_mhp_model(train_data,parent_edge_index,parenttype)

        out=out.to(device)
        #pdt_embeddings=pdt_indexes[pdt_indexes]
        #parent_indexes=edge_label_index[1]
        #parent_indexes=class_indexes.to(device)
        #loss = 1 - F.cosine_similarity(embedding1, embedding2, dim=1).mean()
        train_loss=criterion(out,parent_indexes)
        #loss=loss*100+1

        train_loss.backward()
        optimizer.step()
        losses.append(train_loss)


        predicted_classes = torch.argmax(out, dim=1)  # 
        predicted_classes=predicted_classes.to(device)

        # Compare with true labels
        correct_predictions = (predicted_classes == parent_indexes).sum().item()

        # Calculate accuracy as a percentage
        train_accuracy = correct_predictions / parent_indexes.size(0) * 100
      

        if epoch % 10 == 0:
          #test_acc = test(loader=test_loader, test_model=model,ds=dataset)
          #print("epoch,train_loss",epoch,loss)
          accuracy,loss= evaluate_hgnn_mhp(gnn_mhp_model,val_data,edge_label_index,device,cfg)
          
          test_accs.append(accuracy)
          print("epoch,train_loss,train_accuracy",epoch,train_loss,train_accuracy)
          print("epoch,val loss: val_accuracy",epoch,loss,accuracy)
          #print("val loss: val_accuracy",loss,accuracy)

          if False: #examine graidents
            for name, param in gnn_mhp_model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.norm()}")
                else:
                    print(f"{name}: No gradient")
          
          if accuracy > best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(gnn_mhp_model)
            best_epoch=epoch
        else:
          test_accs.append(test_accs[-1])
    print("best epoch",best_epoch)
    return test_accs, losses, best_model, best_acc
    #return loss.item()

# Evaluation loop
def evaluate_hgnn_mhp(model,data, edge_label_index,device,is_validation=False, save_model_preds=False,config_filename="default"):
    #model=test_model
    
    model.eval()
    data=data.to(device)

    parenttype=cfg.model.pred_parent
    parent_edge_index,data_addnl_edges=get_product_parent_edge_index(data,parenttype=cfg.model.pred_parent)
    if cfg.dataset.add_direct_pdt_edges:
        data=data_addnl_edges
    parent_indexes =parent_edge_index[1]

    parent_indexes=parent_indexes.to(device)
    parent_edge_index=parent_edge_index.to(device)
    

    #pdt_dpt_edge_index=get_product_depts_edge_index(data)

    #parent_indexes=pdt_dpt_edge_index[1]
    

    out = model(data,parent_edge_index,parenttype)
    #pdt_embeddings=pdt_indexes[pdt_indexes]

    criterion = nn.CrossEntropyLoss() 
    
   # class_indexes=edge_label_index[1]  -- replaced  by parent indexes
    #class_indexes=class_indexes.to(device)

    loss=criterion(out,parent_indexes)

    predicted_classes = torch.argmax(out, dim=1)  # 

    # Compare with true labels
    correct_predictions = (predicted_classes == parent_indexes).sum().item()

    # Calculate accuracy as a percentage
    accuracy = correct_predictions / parent_indexes.size(0) * 100

    if save_model_preds:
        print ("Saving Model Predictions for config filename", config_filename)

        data = {}
        data['pred'] = torch.argmax(out, dim=1).cpu().numpy()  #  out.view(-1).cpu().detach().numpy()
        data['label'] = parent_indexes.view(-1).cpu().detach().numpy()

        df = pd.DataFrame(data=data)
        # Save locally as csv 
        current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        #make_dot(out, params=dict(model.named_parameters())).render("model_architecture", format="png")

        df.to_csv('results\itmmh-link-' + config_filename +"-" + current_datetime + '.csv', sep=',', index=False)

#mask = data.val_mask if is_validation else data.test_mask

    return accuracy,loss
        

        # Convert scores to binary predictions





        # Compute F1 score



def train_evaluate_hgnn_mhp(config_filename,config_obj,device):

    cfg=config_obj

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device="cpu"

    split=split=cfg.dataset.split

    ds_data = hItemDataset(root=cfg.dataset.dir,split=split,openai=cfg.dataset.open_ai_embed)

    #ds= [tup[0] for tup in ds_data_graph]

    #mhdata=ds[0]

    # Create a PyTorch Geometric Data object
    #data = Data(x=x, edge_index=edge_index)
    #data=Data(ds_data.data.x,ds_data.data.edge_index)

    #data = train_test_split_edges(data=data,val_ratio=cfg.dataset.valratio)  # Split the edges into train, val, and test sets

    # Train the model

    transform = RandomLinkSplit(
        num_val=split[1],                  # Proportion of edges for validation
        num_test=split[2],                 # Proportion of edges for testing
        is_undirected=False,          # Adjust based on your graph's nature
        add_negative_train_samples=False,  # Add negative samples for training
        edge_types=[('subcats', 'childof', 'cats'),('depts', 'childof', 'subcats'),
                    ('subdepts', 'childof', 'depts'),('classes', 'childof', 'subdepts'),
                    ('products', 'childof', 'classes'),('vendors', 'supply', 'products')]
    )



    train_data, val_data, test_data = transform(ds_data.data)

    
    train_data=train_data.to(device)
    val_data=val_data.to(device)
    test_data=test_data.to(device)


    #test_accs, losses, best_model, best_acc

    test_accs, train_loss,best_model, best_acc = train_hgnn_mhp(train_data=train_data,val_data=val_data,device=device,cfg=cfg)

   

    parent_edge_index,data_addnl_edges=get_product_parent_edge_index(test_data,parenttype=cfg.model.pred_parent)
    if cfg.dataset.add_direct_pdt_edges:
        test_data=data_addnl_edges
    parent_indexes =parent_edge_index[1]

    parent_indexes=parent_indexes.to(device)
    parent_edge_index=parent_edge_index.to(device)


    test_accuracy,test_loss=evaluate_hgnn_mhp(best_model,data=test_data,edge_label_index=parent_edge_index,device=device,save_model_preds=True,config_filename=config_filename)
    #, ds_data,device,is_validation=False, save_model_preds=True,config_filename=config_filename)

    plt.title("MH Node link Prediction for " + config_filename)
    #losses = train_loss #[tensor.item() for tensor in train_loss]
    #losses.to(device)
    #test_accs.to(device)dg
     
    losses=[tensr.detach().cpu().numpy() for  tensr in train_loss]
    plt.plot(losses, label="training loss" + " - " + cfg.model.type)

   # test_accs=[tensr.detach().cpu().numpy() for  tensr in test_accs]
    plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)

    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('results\itmmh-Node-'+ config_filename + "-" + current_datetime + '.png')
    return train_loss,test_accs,test_loss,test_accuracy

def train_gnn_nc(dataset, cfg,device):
 

    # build model
   
    model = GNNStack(dataset.num_node_features, cfg.gnn.dim_inner, dataset.num_classes,cfg,emb=False)
    #self, input_dim, hidden_dim, output_dim, cfg,emb=False
    model=model.to(device)
    #dataset.to(device)
    #scheduler, opt = build_optimizer(args, model.parameters())

    opt= torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)

    # train
    
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(cfg.optim.max_epoch, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        #for batch in loader:
        batch=dataset
        batch=batch.to(device)
        opt.zero_grad()
        pred = model(batch.x,batch.edge_index)
        label = batch.y
        pred = pred[batch.train_mask]
        label = label[batch.train_mask]
        loss = model.loss(pred, label)
        loss.backward()
        opt.step()
        total_loss += loss.item() * 1 #batch.num_graphs
        #total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
          test_acc = test_gnn_nc(test_model=model,ds=dataset)
          test_accs.append(test_acc)
          if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc

def test_gnn_nc(test_model, is_validation=False, save_model_preds=False, model_type=None,ds=None):
    test_model.eval()

    correct = 0
    # Note that Cora is only one graph!
    #for data in loader:
    data=ds
    with torch.no_grad():
        # max(dim=1) returns values, indices tuple; only need indices
        pred = test_model(data.x,data.edge_index).max(dim=1)[1]
        label = data.y

    mask = data.val_mask if is_validation else data.test_mask
    # node classification: only evaluate on nodes in test set
    pred = pred[mask]
    label = label[mask]

    if save_model_preds:
        print ("Saving Model Predictions for Model Type", model_type)

        data = {}
        data['pred'] = pred.view(-1).cpu().detach().numpy()
        data['label'] = label.view(-1).cpu().detach().numpy()

        df = pd.DataFrame(data=data)
        # Save locally as csv 
        current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df.to_csv('results\itmmh-Node-' + model_type + current_datetime + '.csv', sep=',', index=False)

    correct += pred.eq(label).sum().item()

    total = 0
    #for data in loader.dataset:
    data=ds
    total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total

def train_evaluate_gnn_nc(config_filename,config_obj,device):

    cfg=config_obj

    ds_data_graph = ItemDataset(root=cfg.dataset.dir,split=cfg.dataset.split,openai=cfg.dataset.open_ai_embed)

    ds= [tup[0] for tup in ds_data_graph]

    mhdata=ds[0]

    # Create a PyTorch Geometric Data object
    #data = Data(x=x, edge_index=edge_index)
    data=Data(mhdata.x,mhdata.edge_index)

    #data = train_test_split_edges(data=data,val_ratio=cfg.dataset.valratio)  # Split the edges into train, val, and test sets

    #transform = RandomLinkSplit(is_undirected=False, num_val=0.15,num_test=0.15, add_negative_train_samples=True)  

    #train_data, val_data, test_data = transform(data)

    #train_data=train_data.to(device)
    #test_data=test_data.to(device)
    #val_data=val_data.to(device)

    # Train the model
    test_accs, train_losses, best_model, best_acc = train_gnn_nc(mhdata,cfg,device)  #(train_data=train_data,val_data=val_data,device=device,cfg=cfg)
 

    #use the best model to save the scores and plot

    val_accu=test_gnn_nc(best_model,is_validation=True,save_model_preds=True,model_type=cfg.model.type,ds=mhdata) #(model=best_model, data=test_data,device=device,is_validation=False, save_model_preds=True,config_filename=config_filename)

    plt.title("MH Node link Prediction for " + config_filename)
    tlosses =train_losses #[tensor.item() for tensor in train_losses]
    #losses.to(device)
    #test_accs.to(device)dg
    plt.plot(tlosses, label="training loss" + " - " + cfg.model.type)
    plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)

    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('results\itmmh-Node-'+ config_filename + "-" + current_datetime + '.png')
    val_accu=test_gnn_nc(best_model,is_validation=True,save_model_preds=True,model_type=cfg.model.type,ds=mhdata) #(model=best_model, data=test_data,device=device,is_validation=False, save_model_preds=True,config_filename=config_filename)
    return tlosses,test_accs,val_accu

def train_MHGNN(train_data,val_data,device,cfg):
    
    #test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_node_features=train_data.x.shape[1]
    num_classes=1393

    model=MHGNN(input_dim=num_node_features,hidden_dim=cfg.gnn.dim_inner,num_classes=num_classes)
           
    model=model.to(device)
    train_data=train_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)

    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(cfg.optim.max_epoch, desc="Training", unit="Epochs"):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index) 
        pos_neg_score=model.edge_score(train_data.edge_label_index,out,out)  #use both positive and -ve edges
        #loss=model.loss(pos_neg_score,data.edge_label)
        loss = F.binary_cross_entropy_with_logits(pos_neg_score, train_data.edge_label)  #works better than logsoftmax
   
        loss.backward()
        optimizer.step()
        losses.append(loss)
        if epoch % 10 == 0:
          #test_acc = test(loader=test_loader, test_model=model,ds=dataset)
          #print("epoch,train_loss",epoch,loss)
          f1,prec,recall = evaluate_gnn(model=model,data=val_data,device=device)
          test_accs.append(f1)
          print("epoch,train_loss",epoch,loss.cpu().detach().numpy())
          print("test scores: epoch, f1,prec,recall",epoch,loss.cpu().detach().numpy(),round(f1,2),prec,recall)
          
          if f1 > best_acc:
            best_acc = f1
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc
    #return loss.item()

# Evaluation loop
def evaluate_MHGNN(model,data, device,is_validation=False, save_model_preds=False,config_filename="default"):
    #model=test_model
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        # Compute scores for positive and negative edges

        pos_neg_score=model.edge_score(data.edge_label_index,out,out)  #use both positive and -ve edges


        y_pred=pos_neg_score

        y_pred = (y_pred > 0).float()
        y_true=data.edge_label        

 
        data_l=data.edge_label_index
        data_l=torch.cat([data_l,y_true.unsqueeze(dim=0)])
        data_l=torch.cat([data_l,y_pred.unsqueeze(dim=0)])


        # Compute F1 score
        precision = precision_score(y_true.cpu(), y_pred.cpu())
        recall = recall_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        if save_model_preds:
            print ("Saving Model Predictions for config filename", config_filename)

            data = {}
            data['pred'] = y_pred.view(-1).cpu().detach().numpy()
            data['label'] = y_true.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data_l.cpu())
            # Save locally as csv 
            current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
            df.to_csv('results\itmmh-link-' + config_filename +"-" + current_datetime + '.csv', sep=',', index=False)

        #mask = data.val_mask if is_validation else data.test_mask

        return f1,precision,recall
def train_evaluate_MHGNN(config_filename,config_obj,device):

    cfg=config_obj

   

    ds_data_graph = ItemDataset(root=cfg.dataset.dir,split=cfg.dataset.split,openai=cfg.dataset.open_ai_embed)

    ds= [tup[0] for tup in ds_data_graph]

    mhdata=ds[0]

    # Create a PyTorch Geometric Data object
    #data = Data(x=x, edge_index=edge_index)
    data=Data(mhdata.x,mhdata.edge_index)

    #data = train_test_split_edges(data=data,val_ratio=cfg.dataset.valratio)  # Split the edges into train, val, and test sets

    transform = RandomLinkSplit(is_undirected=False, num_val=0.15,num_test=0.15, add_negative_train_samples=True)  

    train_data, val_data, test_data = transform(data)

    train_data=train_data.to(device)
    test_data=test_data.to(device)
    val_data=val_data.to(device)

    # Train the model
    test_accs, train_losses, best_model, best_acc = train_gnn(train_data=train_data,val_data=val_data,device=device,cfg=cfg)
 

    #use the best model to save the scores and plot

    f1_score,prec,recall=evaluate_gnn(model=best_model, data=test_data,device=device,is_validation=False, save_model_preds=True,config_filename=config_filename)

    plt.title("MH Node link Prediction for " + config_filename)
    losses = [tensor.item() for tensor in train_losses]
    #losses.to(device)
    #test_accs.to(device)dg
    plt.plot(losses, label="training loss" + " - " + cfg.model.type)
    plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)

    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('results\itmmh-Node-'+ config_filename + "-" + current_datetime + '.png')
    return f1_score,prec,recall,losses,test_accs

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


