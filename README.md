# Data Mining and Data Warehousing Project
Repo for Data Mining and Data Warehousing project which consists of a pipeline for obtaining clusters on 2 datasets (market segmentation task)

## Project structure:
* data/ - location of the data
  * dataset_list.txt - contains the URL to the two datasets used
* scripts/ - location where all the scripts used are stored
  * datasets.py - python file with multiple dataset classes for each dataset type
  * eda.py - python script that runs different EDA statistics
  * main.py - python script that runs the OPTICS clustering algorithm on the datasets
  * utils.py - python file with useful functions
