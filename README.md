# go_with_the_flows
Official implementation of "Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generationand Reconstruction"

# Environment


# Datasets
Preprocessed data are provided on

Or you can use scripts preprocess_ShapeNetAll.py for processing ShapeNetAll13 dataset

  python preprocess_ShapeNetCore.py data_dir save_dir

and split the data into train/val/test sets:
  
  python resample_ShapeNetCore.py data_path
  
preprocess_ShapteNetCore.py for procesing ShapeNetCore55 dataset
  
  python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir
  
# Training

All pretrained models including seperate configs can be downloaded on
To use the models, you need to download the models and put the files in the code directory.
Then specify the path2data storing preprocessed data and path2save directory storing all saved
checkpoints.

For class-conditional generative models, run:
  
  ./scripts/train_airplane_gen.sh
  
  ./scripts/train_car_gen.sh
  
  ./scripts/train_chair_gen.sh

For auto-encoding task, run:

  ./scripts/train_all_ae.sh

For SVR task, run:

  ./scripts/train_all_svr.sh

# Evaluation
When using pretrained models, evaluation can be done by runing:

For generation task, take airplanes category as an example:
 
  ./scripts/run_evaluate_gen.sh

For auto-encoding task:

  ./scripts/run_evaluate_ae.sh

For SVR task,

  ./scripts/run_evaluate_svr.sh

# Visualization

For visualization with Mitsuba Renderer, we need to first install Mistsuba 2.0 following its document. Then we run:

  ./scripts/render.sh
 
to generate the rendered figures.





