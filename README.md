# Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction
This is the official implementation of [Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction](https://arxiv.org/abs/2106.03135).

This repository is based on the official implementation of [Discrete Point Flow](https://github.com/Regenerator/dpf-nets).

# Environment

This repository requires:

- pytorch
- ...

We further provide all necessary requirements in for of a `requirements.txt`.

# Datasets

We train our models on [ShapeNet](https://shapenet.org/). Specifically, we use [ShapeNetCore55](https://www.shapenet.org/download/shapenetcore) in our experiments on generative modeling and autoencoding, 
and [ShapeNetAll13](http://3d-r2n2.stanford.edu/) in the ones on single-view reconstructions. After downloading, the datasets can be preprocessed by running:

```python preprocess_ShapeNetCore.py data_dir save_dir```

resp.

```python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir```

Subsequently, for ShapeNetCore55 `train/val/test` splits are created using:

```python resample_ShapeNetCore.py data_path```

## Download preprocessed datasets

Since the preprocessing takes up to a week, we provide the preprocessed datasets:

- [Preprocessed ShapeNetCore55](https://data.vision.ee.ethz.ch/jpostels/go_with_the_flows/ShapeNetCore55v2_meshes_resampled.h5)
- Preprocessed ShapeNetAll13 [meshes](https://data.vision.ee.ethz.ch/jpostels/go_with_the_flows/ShapeNetAll13_meshes.h5) and [images](https://data.vision.ee.ethz.ch/jpostels/go_with_the_flows/ShapeNetAll13_images.h5).


# Pretrained models

All pretrained models including the corresponding config files can be downloaded [here](https://drive.google.com/drive/folders/1fkVBVqxy2_zTevwd3WdnROPreYke-zuU?usp=sharing).
To use the models, you need to download the models and put the files in the root directory `./`.
Then, specify the `path2data` storing preprocessed data and path2save directory storing all saved
checkpoints.
  
# Training

All training configurations can be found in `configs/`. Prior to training/evaluation remember to set
`path2data` in the resp. config file accordingly.

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

For visualization with Mitsuba Renderer, we need to first install and compile Mistsuba 2.0 following the [official documentation](https://www.mitsuba-renderer.org/). Note, mitsuba needs to be sourced before using it every time. Then run `evaluate_ae.py` with flag `--save` to generate the `.h5` file consisting
of ground-thruth point clouds and corresponding generated point clouds. Subsequently, point clouds can be rendered by running:

```
bash ./scripts/render.sh
```
where `path_h5` denotes the path of `.h5` file, `path_png` denotes the path to save png files, `path_mitsuba` represents the path where mitsuba can be used, `name_png` represents the name to save png files, `indices` can be put with all indexes of samples that you want to render. All these strings need to be specified.

# Citation

```
@article{postels2021go,
  title={Go with the Flows: Mixtures of Normalizing Flows for Point Cloud Generation and Reconstruction},
  author={Postels, Janis and Liu, Mengya and Spezialetti, Riccardo and Van Gool, Luc and Tombari, Federico},
  journal={International Conference on 3D Vision},
  year={2021}
}
```





