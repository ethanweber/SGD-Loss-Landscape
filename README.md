# SGD-Loss-Landscape


#### <hr> Setup Environment

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