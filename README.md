# Augmented_CGCNN

This software package implements a data augmentation method to enable machine learning models to predict the relaxed formation energy of unrelaxed structure inputs. 

The major function of the package is to augment the training and validation data used to train a machine learning model.

The following paper describes the augmentation technique:
[Data-Augmentation for Graph Neural Network Learning of the Relaxed Energies of Unrelaxed Structures](https://arxiv.org/abs/2202.13947)

The software used to train the [CGCNN](https://github.com/txie-93/cgcnn) and the [CGCNN-HD](https://github.com/kaist-amsg/CGCNN-HD) are well documented in the respective respitory.

## Files and directories
`pre-trained/` contains the files needed to load all models used in the study.


`test_data/unrelaxed/` and `test_data/relaxed/` contain Test-unrelaxed and Test-relaxed. In each directory the `atom_init.json` file contains embedding information for the CGCNN. The `id_prop.csv` is the file needed for the CGCNN to load the data. The first column is the structure id the second column is the relaxed formation energy per atom of the structure.


`cgcnn/data.py` and `cgcnn/model.py` contain dataloaders and the CGCNN, respectively.


`cgcnn/model_train.py` contains the class `MPCrystalGraphConvNet` which places each convolutional layer on a seperate GPU. This helps alleviate memory issues when a batch contains multiple structures with more than 64 atoms.


`augment_mp.py` queries MaterialsProject and writes the training and validation datasets.


`dists.pkl` contains the ditribution used to perturb the structures.


`predict.py` predicts the formation energy per atom of both Test-relaxed and Test-unrelaxed using all four models from the study.



##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `augmented_cgcnn` and install all prerequisites:

```bash
conda upgrade conda
conda create -n augmented_cgcnn python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
```

This creates a conda environment for running CGCNN and perturbing structures. Activate the environment by:

```bash
source activate augmented_cgcnn
```
## How to augment data from [MaterialsProject](https://materialsproject.org)
to generate the augment training and validation set; add your materialsproject API key to `augment_mp.py` and run:

```bash
mkdir train_data
mkdir validation_data
python augment_mp.py
```
This will perturb every structure in the MP database and write 80% of the data (which consists of one perturbed structure for every relaxed structure) to train_data and 20% to validation_data.

## How to make a prediction using the pretrained models
to make predictions on Test-relaxed and Test-unrelaxed, run:
```bash
python predict.py
```
This will print the prediction errors for the four models and write the predictions and target values to a csv file. The first column is the DFT value, the second column is the predicted value.
