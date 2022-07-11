# Augmented_CGCNN

This software package implements a data augmentation method to enable machine learning models to predict the relaxed formation energy of unrelaxed structure inputs. 

The major function of the package is to augment the training and validation data used to train a machine learning model.

The following paper describes the augmentation technique:
[Data-Augmentation for Graph Neural Network Learning of the Relaxed Energies of Unrelaxed Structures](https://arxiv.org/abs/2202.13947)

The software used to train the [CGCNN](https://github.com/txie-93/cgcnn) and the [CGCNN-HD](https://github.com/kaist-amsg/CGCNN-HD) are well documented in the respective respitory.

## How to augment data from [MaterialsProject](https://materialsproject.org)
to generate the augment training and validation set; add your materialsproject API key to `augment_mp.py` and run:

```bash
mkdir train_data
mkdir validation_data
python augment_mp.py
```
This will perturb every structure in the MP database and write 80% of the data (which consists of one perturbed structure for every relaxed structure) to train_data and 20% to validation_data.

## How to make a prediction using the pretrained models

