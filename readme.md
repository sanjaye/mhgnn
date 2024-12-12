config files are in config folder
data folder should have two csv files (all_nodes_.... and d_all_edges)

all_nodes###.csv has the list of all nodes in the following format
        "nodeid","orignodeid","nodename","lvl"
        0,2362,ROOT,0
        1,2363,SUPPLIES,1

        nodeid+ sequential number used in the datasets 0 indexed
        
        In the file we can just have a sequential list of nodeids.In hetrogeneous datasets, nodeid is reset to 0-starting for all node types (for example dept and subdept both will have nodeid=0).  This is done in the code (hetroitmdataset.py)
        orignodeid- nodeid that is used in the business database (external system)
        nodename- name of the merchandise hierarchy node
        lvl=  level of the node.  There are 7 levels (0-cat,1-subcat,2-dept,3-subdept,4-class,5-product,6-vendor)
d_all_edges###.csv has the following format
        "source","target","edgetype"
        0,1,0
        0,2,0
        source = starting nodeid
        target= destination nodeid
        edige type 0 for cat-subcat link 1 for subcat-dept link etc.
        This is set up as a hierarchical/directed graph
Code files
   cmd_args.py -  commandline arguments (none are eventually used )
   config.py:  config object (representing the various yaml files)
   mhgnns.py -  Most important file.  It contains all the classes, 
   mhgnntraineval training loops and evaluation loops
   itmdataset.py -  used in graphsage and GAT experiemnts
   hetroitmdataset.py - used in hetrognn experiments
   run.py -   This is the programto run. It will loop through the config files in the config
   folder and find the best model/config for link-prediction (coded in hetrographlinks.py)     


