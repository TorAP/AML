# Generating Monet Paintings via CycleGAN
This repository contains code to train a CycleGAN and analyze the results.

-----------
## Data 
- The dataset used was downloaded from Kaggle
and contains two directories: Monet and Photo. 
- The Monet
directory contains 300 Monet paintings sized 256x256 in JPEG
format.
- The Photo directory contains 7028 photos sized
256x256 in JPEG format. 
--------

## Use pre-trained model
- The notebook "GenerateMonetFromExcistingModel.ipynb" uses google drive to upload the data. In order for that notebook to run smoothly, uploading a zip of the generate.zip folder to your google drive is necessary. Here is a link to the folder: https://drive.google.com/file/d/1NZMds1WLYqjvS8Qrhb3fIaOAhrWGK7XJ/view?usp=sharing
- The runbook generates Monets from 4 different pictures using a pre-existing model (the best model) for two different epoch numbers (epoch 2 and epoch 100). These results are then combined into one .png image called "GeneratedMonets.png". 
--------
## To train model: 
- Running main.py will train a model
- To tweak parameters you can select a subset of hyperparamters, by chaning the 
  index in the training step. 
- To monitor performance you can add your own Weight and Biasis api-key. 
- 6 types of images will be generated during training:
  - Original photos
  - Generated Monet
  - Reconstructed 
  - Original Monet
  - Generated photo
  - Reconstructed Monet


-------
## Analysis
- Data for the best model can be downloaded: https://drive.google.com/file/d/115vhzf6M-uzOwk8RGOGQh5i3a4YOnCUZ/view?usp=share_link
  - The data has already been pre-processed (so this step can be skipped in the analysis)
- The analysis conists of: 
  - A RGB comparison of images from a previous training with 3 different metrics 
    (MSE,PSNR,SSIM)
  - A grey-scale comparison of images from a previous training
  - A comparison of original photos and reconstructed photos 
