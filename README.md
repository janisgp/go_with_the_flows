# Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction
This is the official implementation of [Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction](https://arxiv.org/abs/2106.03135)

# Environment

This repository requires:

- pytorch
- ...

We further provide all necessary requirements in for of a `requirements.txt`.

# Datasets
Our preprocessed dataset can be downloaded [here](link.dataset)

Alternatively `preprocess_ShapeNetAll.py` can be used to process ShapeNetAll13 dataset. Therefore, first download [ShapeNetAll13](Shapenet.link). Subsequently, the dataset can be preprocessed using:
  
```python preprocess_ShapeNetCore.py data_dir save_dir```

Then, train/val/test splits can be created using:
  
```python resample_ShapeNetCore.py data_path```
  
preprocess_ShapteNetCore.py for procesing ShapeNetCore55 dataset
  
  python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir

# Pretrained models

All pretrained models including the corresponding config files can be downloaded [here](pretrained_models.link).
To use the models, you need to download the models and put the files in the root directory `.`.
Then, specify the path2data storing preprocessed data and path2save directory storing all saved
checkpoints.
  
# Training

## Generative modeling

A generative model can be trained on airplanes/cars/chairs by running the corresponding subsequent command:
 
``` 
bash ./scripts/train_airplane_gen.sh
  
bash ./scripts/train_car_gen.sh
  
bash ./scripts/train_chair_gen.sh
```

## Autoencoding

An Autoencoder on the entire ShapeNet dataset can be trained using:

```
bash ./scripts/train_all_ae.sh
```

## Single-view reconstruction

To train our model on single-view reconstruction, run:

```
bash ./scripts/train_all_svr.sh
```

# Evaluation

## Generative modeling

Generative models can be evaluated by running:
 
```
./scripts/run_evaluate_gen.sh
```

## Autoencoding

Autoencoding can be evaluated by running:

```
./scripts/run_evaluate_ae.sh
```

## Single-view reconstruction

Single-view reconstruction can be evaluated by running:

```
./scripts/run_evaluate_svr.sh
```

# Visualization

For visualization with Mitsuba Renderer, we need to first install Mistsuba 2.0 following the [official documentation](link.to.offical.documentation). Subsequently, point clouds can be rendered by running:

```
./scripts/render.sh
```

# Citation

```
@article{postels2021go,
  title={Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction},
  author={Postels, Janis and Liu, Mengya and Spezialetti, Riccardo and Van Gool, Luc and Tombari, Federico},
  journal={International Conference on 3D Vision},
  year={2021}
}
```





