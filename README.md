# TODO
- if mainhydra.py works for both, we can remove mainhydra_turbolence.py (and maybe just rename to main.py)
- Once the dataset is ready, modify the following functions:
    - postprocessing/plots.py: write plot_predictions1D and plot_prediction2D and delete plot_predictions_legacy
    - mainhydra.py: remove lines 50-56 if possible
    - utils/variable_generator.py: remove all file
    - config folder: modify data folder


# PACSproject

To use the *visualization_turbulent_radiative_layer_2D* dataset, you need to install in your environment:

```
pip install the_well
```

and then:

```
the-well-download --base-path path/to/base --dataset turbulent_radiative_layer_2D --split train
```
where path/to/base should be the path you want it to be installed in (starting from current path)

IDEAS:
Cleanliness of the code:
- add abstract classes targetnetwork and hypernetwork? This would be useful to include parameters like replace_weights
- testing function replicates code of training function
- add something ready to use for inference?
Future development:
- use a graph structure (ask Riccardo) for connecting different modules of hypernetwork and targetnetwork
- split hypernetwork in backbone and head:
    -backbone acts as feature extractor: takes hyperparameters and produces a latent space
    -head maps the latent space to the weights of the targetnetwork: we can design different heads with different initializations straegies to match the needs of the targetnetworks