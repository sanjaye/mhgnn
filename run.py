import os
import yaml
import json  
 
import torch
import hetroitmdataset
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
import datetime
from cmd_args import parse_args
from config import cfg,load_cfg,set_out_dir,dump_cfg
from mhgnntraineval import train_evaluate_hgnn,train_evaluate_gnn,train_evaluate_hgnn_mhp,train_evaluate_gnn_nc
#import graphlinks #import train_evaluate

def load_config(file_path):
    """
    Load a YAML configuration file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_configs(config_dir, log_file=".\results\evaluation_results.json",cfg=None):
    """
    Evaluate multiple configurations and find the best one.

    Args:
        config_dir (str): Directory containing all configuration YAML files.
        log_file (str): Path to save evaluation results as JSON.

    Returns:
        tuple: Best configuration file and its score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device="cpu"
    
    best_config = None
    best_score = float('-inf')  # Initialize with the lowest possible score
    results = []


    

    # Loop through all configuration files in the directory
    for config_file in os.listdir(config_dir):
        if config_file.endswith(".yml") or config_file.endswith(".yaml"):
            config_path = os.path.join(config_dir, config_file)

            args = parse_args()
            args.cfg_file=config_path
            # Load config file
            load_cfg(cfg, args)

            set_out_dir(cfg.out_dir, args.cfg_file)
            # Set Pytorch environment
            torch.set_num_threads(cfg.num_threads)
            
            dump_cfg(cfg)
           # cfg = load_config(config_path)
            #best_score=0
            print(f"Evaluating configuration: {config_file}")
            if cfg.dataset.task=="node":  # node classification
                 
                tlosses,test_accs,val_accu = train_evaluate_gnn_nc(config_file,cfg,device)

                print("Maximum test set accuracy: {0}".format(max(test_accs)))
                print("Minimum train loss: {0}".format(min(tlosses)))

                minloss=min(tlosses)
                maxtestacc=max(test_accs)

                results.append({"config_file": config_file, "train_loss":  minloss,"test_acc":maxtestacc,"val_accuracy":val_accu})

                plt.title("MH Node Type Prediction")
                plt.plot(tlosses, label="training loss" + " - " + cfg.model.type)
                plt.plot(test_accs, label="test accuracy" + " - " + cfg.model.type)
                current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig('results\itmmh-Node-'  + current_datetime + '.png')

                if val_accu > best_score:
                    best_score = best_score
                    best_config = config_file
                log_file="results\eval_results_NC_" + current_datetime +".json"

            elif cfg.model.type=="MH_HETRO":  #Predit MH Parents using Hetro GNN
                train_loss,train_accuracy,test_loss,test_accuracy=train_evaluate_hgnn_mhp(config_file,cfg,device)
                print()
                print(f"Train Results for  configuration: {config_file}")
                print("Maximum (training time) test set accuracy: {0}".format(max(train_accuracy)))
                print("Minimum train loss: {0}".format(min(train_loss)))

                print(f"Best Test Results for  configuration: {config_file}")
                print("Best test loss and accuracy:",test_loss,test_accuracy)

                tlosses=[tensr.detach().cpu().numpy() for  tensr in train_loss]
                minloss= float(min(array.min() for array in tlosses))  #avoid json non-serializabe error

            
                # Log the result
               
                results.append({"config_file": config_file, "train_loss":  minloss,"train_accuracy":max(train_accuracy),"test_loss":test_loss.detach().item(),"test_accuracy":test_accuracy})

                # Update the best configuration
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_config = config_file
                #results=results.tolist()
                current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file="results\eval_results_MHP_" + current_datetime +".json"
            else:
                if cfg.model.type=="MHGNN":
                    pass
                elif cfg.model.type=="HETRO":  # Use HETRO GNN for link prediction
                    f1_score,prec,recall,train_losses,train_test_accs=train_evaluate_hgnn(config_file,cfg,device)
                else:   #Predict links using GAT/GraphSage
                    f1_score,prec,recall,train_losses,train_test_accs=train_evaluate_gnn(config_file,cfg,device)
                current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file="results\eval_results_LINK_" + current_datetime +".json"
                       
                print()

                
                print(f"Train Results for  configuration: {config_file}")
                print("Maximum (training time) test set accuracy: {0}".format(max(train_test_accs)))
                print("Minimum train loss: {0}".format(min(train_losses)))

                print(f"Best Test Results for  configuration: {config_file}")
                print("Best Scores F1, Prec, Recall:",f1_score,prec,recall)
            
                # Log the result
                results.append({"config_file": config_file, "score": f1_score,"prec":prec, "recall":recall})

                # Update the best configuration
                if f1_score > best_score:
                    best_score = f1_score
                    best_config = config_file

    # Save results to a JSON file
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Best configuration: {best_config} with score: {best_score}")
    return best_config, best_score

if __name__ == "__main__":
    # Directory containing configuration files
    config_directory = "./configs"  # Replace with your actual directory

    #config_directory="C:\\cs224w_code\\pyggym\\graphsage_b1\\config"

    config_directory="./config"

    # Call the evaluation function #cfg is populated from import, but will be rewritten with the args.cfg-file in the loop
    best_config, best_score = evaluate_configs(config_directory,log_file="eval_results.json",cfg=cfg)

    # Print the final results
    print(f"The best configuration is '{best_config}' with a score of {best_score}.")
