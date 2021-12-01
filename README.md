# go_with_the_flows
Official implementation of "Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generationand Reconstruction"

#Environment


#Datasets
Preprocessed data are provided on

Or you can use scripts preprocess_ShapeNetAll.py for processing ShapeNetAll13 dataset

  python preprocess_ShapeNetCore.py data_dir save_dir

and split the data into train/val/test sets:
  
  python resample_ShapeNetCore.py data_path
  
preprocess_ShapteNetCore.py for procesing ShapeNetCore55 dataset
  
  python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir
  
#Training



