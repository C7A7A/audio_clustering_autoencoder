# Clustering audio files with classic algorithms and autoencoder

## Overview
Based on this project I wrote my master thesis (written in polish): https://www.overleaf.com/read/sfmhhkfhbwhy#d03529 

The study analysed the performance of different clustering algorithms on three diverse data sets. The methods operated on audio files that were represented by five audio features: spectrogram, mel-spectrogram, chromagram, MFCCs and MFCCs without the first cepstral coefficient. The research included classical clustering algorithms such as K-Means, Fuzzy C-Means, hierarchical agglomerative clustering and DBSCAN. In addition, a deep convolutional autoencoder architecture for clustering data was implemented. The paper describes in detail the processing and preparation of the audio files, which included the conversion of the raw audio data into suitable feature representations and their further preparation for analysis. Experimental results provide insight into the effectiveness of different clustering methods and their impact on the quality and precision of audio data clustering. 

## Table of Contents
- [Introduction](#introduction)
- [Build with](#build-with)
- [Getting started](#getting-started)
- [Datasets](#datasets)
- [Folders and files](#folders-and-files)
- [Autoencoder](#autoencoder)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In the field of audio signal processing, data analysis is growing in importance. Nowadays, the intensive development of technology makes automatic clustering of audio data crucial in many applications that are used to recognise music, speakers or identify environmental sounds. Recently, various neural network architectures, including autoencoders, have increasingly been used for the task of cluster analysis. For this reason, it is worth to analyse what results can be obtained with autoencoder and how these results differ from those obtained with various classical algorithms.

## Build with
[![Python][Python]][Python-url]
[![Librosa][Librosa]][Librosa-url]
[![Scipy][Scipy]][Scipy-url]
[![Sklearn][Sklearn]][Sklearn-url]
[![Pytorch][Pytorch]][Pytorch-url]
[![Pytorch-Lightning][Pytorch-Lightning]][Pytorch-Lightning-url]
[![Matplotlib][Matplotlib]][Matplotlib-url]

## Getting started
To get started, clone the repository and install the required dependencies (with conda):
```bash
git clone git@github.com:C7A7A/audio_clustering_autoencoder.git
cd audio_clustering_autoencoder
conda create --name myenv python=3.11.5
conda activate myenv
pip install -r requirements.txt
jupyter notebook
```
After that you should be ready to use all scripts located in /notebooks directory.

## Datasets
Three disparate datasets were selected to analyze the clustering results: Music Audio Benchmark Data Set, UrbanSound8K and Toronto Emotional Speech Set. Each of these sets comes from a different data family (music, environmental sounds, speakers), allowing for extensive testing and evaluation of the effectiveness of the clustering methods used. In addition, the MNIST dataset was used to verify that the autoencoder architecture used in the experiments was properly implemented.

#### [Music Audio Benchmark Data Set][MABDS]
Dataset containing 1886 10-second music tracks encoded in mp3 format. The collection of audio files was downloaded from www.garageband.com. The music tracks are divided into 9 categories (alternative, blues, electronic, folkcountry, funksoulrnb, jazz, pop, hiphop and rock).

| **Category**     | **Sample** |
|-------------------|-------------------|
| alternative       | 145               |
| blues             | 120               |
| electronic        | 113               |
| folkcountry       | 222               |
| funksoulrnb       | 47                |
| jazz              | 319               |
| pop               | 116               |
| raphiphop         | 300               |
| rock              | 504               |
| **Total**          | **1886**           |

*Number of songs in different music genres*

#### [UrbanSound8K][US8K]
Dataset containing 8732 tagged audio files, which are urban sounds divided into 10 classes (air conditioner, car horn, children playing, dog barking, drill drilling, engine, gun shot, jackhammer, siren and street music). All data are from field recordings sent to www.freesound.org.

| **Category**       | **Sample** |
|---------------------|-------------------|
| air_conditioner     | 1000              |
| car_horn            | 429               |
| children_playing    | 1000              |
| dog_bark            | 1000              |
| drilling            | 1000              |
| engine_idling       | 1000              |
| gun_shot            | 374               |
| jackhammer          | 1000              |
| siren               | 929               |
| street_music        | 1000              |
| **Total**            | **8732**          |

*Number of environmental sound files by category*

#### [Toronto Emotional Speech Set][TESS]
Dataset created by Kate Dupuis and M. Kathleen Pichora-Fuller at the University of Toronto. The dataset contains 2,800 audio files that contains text read by two actresses. The women read 200 words in seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, neutrality). The spoken phrase is “Say the word _” where _ stands for one of the 200 words. For this dataset, in addition to grouping the data into 14 clusters (which is based on the number of unique labels), the data were also grouped into 7 clusters (the number of unique emotions in the labels).

| **Category**              | **Sample** |
|----------------------------|-------------------|
| OAF_Fear                   | 200               |
| OAF_Pleasant_surprise       | 200               |
| OAF_Sad                    | 200               |
| OAF_angry                  | 200               |
| OAF_disgust                | 200               |
| OAF_happy                  | 200               |
| OAF_neutral                | 200               |
| YAF_angry                  | 200               |
| YAF_disgust                | 200               |
| YAF_fear                   | 200               |
| YAF_happy                  | 200               |
| YAF_neutral                | 200               |
| YAF_pleasant_surprised      | 200               |
| YAF_sad                    | 200               |
| **Total**                   | **2800**          |

*Number of files by actress and emotion*


## Folders and files

## Autoencoder

## Results
The results of the clustering experiments will be saved in the `results/` directory. You can visualize the clustering performance using the provided Jupyter notebooks.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Librosa]:https://img.shields.io/badge/-Librosa-4d02a2?style=for-the-badge
[Librosa-url]: https://librosa.org/doc/latest/index.html
[Scipy]: https://img.shields.io/badge/-SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[Scipy-url]: https://scipy.org/
[Sklearn]: https://img.shields.io/badge/scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Sklearn-url]: https://scikit-learn.org/stable/
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Pytorch-Lightning]: https://img.shields.io/badge/-Pytorch%20Lightning-7544e4?style=for-the-badge
[Pytorch-Lightning-url]: https://lightning.ai/docs/pytorch/stable/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge
[Matplotlib-url]: https://matplotlib.org/

[MABDS]: https://www-ai.cs.tu-dortmund.de/audio.html
[US8K]: https://urbansounddataset.weebly.com/urbansound8k.html
[TESS]: https://tspace.library.utoronto.ca/handle/1807/24487
