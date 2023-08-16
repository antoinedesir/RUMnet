## Description

This is the code associated with the paper *Representing Random Utility Choice Models with Neural Networks* (Aouad and Désir, 2023). The code has two parts.

- Python code base for training RUMnets.
- Python code to run the benchmarks from the paper.

This is a *work-in-progress* repository.

## Usage

The modules in `src/` contain the experiment auxiliary functions (`experiment_auxiliary.py`) and the Keras model classes (`models.py`).

The script `run_template.py` runs experiments for a given dataset (specify `test_name = "swiss_metro"`) and model class (specify `model_family = "RUMnet"`), which include RUMnet, Tastenet, DeepMNL, random forests (RF), and neural networks (NN).

The configurations of experiments and models should be specified in the config JSON files in the subfolder  `src/test_name/`. These include the general and experiment-specific parameters in the files `config_experiment.json` and `config_general.json` as well as model-specific parameters in the files `config_modelsArchitecture.json`, located in each model subfolder of `src/test_name/modelArchitecture/`.

For new dataset applications, one need to construct a new data_preparation function under the same template format as in `experiment_auxiliary.py` and import the function in the script `run_template.py`.

The results are saved in a folder `output/test_name/`.

## Dependencies

The code was tested with the following:

- python 3.8.13
- numpy 1.19.2
- pandas 1.4.2
- tensorflow 2.4.0 (tf-nightly)
- joblib 0.17.0

The full environment dependencies are provided in `/env/rumnet_env.yml`.

## Data
To run the code, the datasets should be copied into a new subfolder `data/testname` and loaded in the data_preparation function (see above). The Swissmetro dataset is available [here](https://transp-or.epfl.ch/pythonbiogeme/examples_swissmetro.html) and the Expedia dataset is available [here](https://www.dropbox.com/sh/3at79kbztjittvk/AACykfcWhewRqiErmDjrp5Nxa?dl=0).

## License
This project is licensed under the terms of the [MIT License](LICENSE).
