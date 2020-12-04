# SGD-Loss-Landscape

#### <hr> Resources

- [Project Materials](https://drive.google.com/drive/u/0/folders/1zBMinqbImwJ4SZhaPFEqyUttvHNigS-Y)

#### <hr> Setup environment

```
# clone
git clone --recurse-submodules git@github.com:ethanweber/SGD-Loss-Landscape.git

# conda environment
conda create -n landscape python=3.8.2
conda activate landscape
pip install -r requirements.txt

# jupyter
python -m ipykernel install --user --name landscape --display-name "landscape"
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
```

#### <hr> Structure

```
|- configs
    |- <config_name>
|- datasets
    |- <dataset_name>
|- models
    |- <config_name>
|- plots
    |- <dataset_name>
        |- <config_name>
```

#### <hr> Getting started

```
# TODO: use - not _ in args

# create dataset
python generate_dataset.py \
    --n 1000 \
    --d 100 \
    --dataset_name example_dataset

# plot dataset
python plot_dataset.py --dataset_name example_dataset

# create model config
python make_model_configs.py \
    --num-layers 4 \
    --batch-size 1 \
    --dataset-name example_dataset \
    --config-name example_model

# train the model
python run_network.py \


# visualize loss landscape
python plot_landscape.py
```