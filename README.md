# Clustering audio files with classic algorithms and autoencoder

## Overview
Based on this project I wrote my master thesis (written in polish): https://www.overleaf.com/read/sfmhhkfhbwhy#d03529 

The study analysed the performance of different clustering algorithms on three diverse data sets. The methods operated on audio files that were represented by five audio features: spectrogram, mel-spectrogram, chromagram, MFCCs and MFCCs without the first cepstral coefficient. The research included classical clustering algorithms such as K-Means, Fuzzy C-Means, hierarchical agglomerative clustering and DBSCAN. In addition, a deep convolutional autoencoder architecture for clustering data was implemented. The paper describes in detail the processing and preparation of the audio files, which included the conversion of the raw audio data into suitable feature representations and their further preparation for analysis. Experimental results provide insight into the effectiveness of different clustering methods and their impact on the quality and precision of audio data clustering. 

## Table of Contents
- [Introduction](#introduction)
- [Build with](#build-with)
- [Getting started](#getting-started)
- [Datasets](#datasets)
- [Files and folder structure](#files-and-folder-structure)
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
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/audio_clustering_autoencoder_versus_classic.git
cd audio_clustering_autoencoder_versus_classic
pip install -r requirements.txt
```

## Datasets

## Files and folder structure

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
[Matplotlib]: https://img.shields.io/badge/-Matplotlib-000000?style=for-the-badge
[Matplotlib-url]: https://matplotlib.org/