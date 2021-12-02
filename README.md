# Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction
This is the official implementation of [Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction](https://arxiv.org/abs/2106.03135)

# Environment

This repository requires:

- pytorch
- ...

We further provide all necessary requirements in for of a `requirements.txt`.

# Datasets

`preprocess_ShapeNetAll.py` can be used to process ShapeNetAll13 dataset. Therefore, first download [ShapeNetAll13](https://shapenet.org/). Subsequently, the dataset can be preprocessed using:
  
```python preprocess_ShapeNetCore.py data_dir save_dir```

Then, `train/val/test` splits can be created using:
  
```python resample_ShapeNetCore.py data_path```
  
`preprocess_ShapteNetCore.py` for procesing [ShapeNetCore55](https://shapenet.org/) dataset
  
```python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir```


# Pretrained models

All pretrained models including the corresponding config files can be downloaded [here](https://drive.google.com/drive/folders/1fkVBVqxy2_zTevwd3WdnROPreYke-zuU?usp=sharing).
To use the models, you need to download the models and put the files in the root directory `.`.
Then, specify the `path2data` storing preprocessed data and path2save directory storing all saved
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
bash ./scripts/run_evaluate_gen.sh
```

## Autoencoding

Autoencoding can be evaluated by running:

```
bash ./scripts/run_evaluate_ae.sh
```

## Single-view reconstruction

Single-view reconstruction can be evaluated by running:

```
bash ./scripts/run_evaluate_svr.sh
```

# Visualization

For visualization with Mitsuba Renderer, we need to first install Mistsuba 2.0 following the [official documentation](https://www.mitsuba-renderer.org/). Then run `evaluate_ae.py` with flag `--save` to generate the `.h5` file consisting
of ground-thuth point clouds and corresponding generated point clouds. Subsequently, point clouds can be rendered by running:

```
bash ./scripts/render.sh
```
where `path_h5`, `path_png`, `path_mitsuba`, `name_png` needs to be specified.

# Citation

```
@article{postels2021go,
  title={Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction},
  author={Postels, Janis and Liu, Mengya and Spezialetti, Riccardo and Van Gool, Luc and Tombari, Federico},
  journal={International Conference on 3D Vision},
  year={2021}
}
```





