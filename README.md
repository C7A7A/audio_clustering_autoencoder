# Clustering audio files with classic algorithms and autoencoder

## Overview
Based on this project I wrote my master thesis (written in polish): https://www.overleaf.com/read/sfmhhkfhbwhy#d03529 

The study analysed results obtained from various clustering algorithms on three diverse data sets. The methods operated on audio files that were represented by five audio features: spectrogram, mel-spectrogram, chromagram, MFCCs and MFCCs without the first cepstral coefficient. The research included classical clustering algorithms such as K-Means, Fuzzy C-Means, hierarchical agglomerative clustering and DBSCAN. In addition, a deep convolutional autoencoder architecture for clustering data was implemented. The paper describes in detail the processing and preparation of the audio files, which included the conversion of the raw audio data into suitable feature representations and their further preparation for analysis. Experimental results provide insight into the effectiveness of different clustering methods and their impact on the quality and precision of audio data clustering. 

![AE](/images-readme/mel_spectrogram_color_grayscale.png)
*Mel-spectrogram in color and grayscale. One of the features extracted to cluster audio data*

## Table of Contents
- [Introduction](#introduction)
- [Build with](#build-with)
- [Getting started](#getting-started)
- [Datasets](#datasets)
- [Directories and files](#directories-and-files)
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
cd notebooks
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

## Directories and files
Directories in the project are used for:
<ul>
    <li> <b> notebooks </b> - To store jupyter notebook scripts. </li>
    <li> <b> datasets </b> - To store datasets. Each dataset should be in a separate directory inside the <b> datasets </b> directory. </li>
    <li> <b> metadata </b> - To store metadata extracted from dataset file and directory structure. </li>
    <li> <b> features </b> - To store features extracted from audio files. </li>
    <li> <b> checkpoints </b> - To store best autoencoder checkpoints. </li>
    <li> <b> results </b> - To store images showing the results of clusterings. </li>
    <li> <b> logs </b> - To store pytoch-lighting logs. </li>
</ul>

Inside <b> notebooks </b> directory there are 6 jupyter notebooks and 1 python file:
<ul>
    <li> <b> 01_save_metadata.ipynb </b> - This script saves relevant information about the dataset based on the directory structure of the dataset and the audio files themselves. The stored metadata included four elements: file name, file extension, path to the file, and a parent folder that served as a label. </li>
    <li> <b> 02_transform_audio.ipynb </b> - This script uses previously prepared metadata to identify the files and convert them to a uniform WAV format and limit the frequency to a defined threshold. Conversion to WAV format is desirable because the librosa library works best with data with this extension. This audio file format is lossless and suitable for further data processing and analysis. Filtering of audio signals helped eliminate unwanted frequencies. A low-pass filter was used to limit the bandwidth of the signal to a preset threshold. Limiting the frequency allows to focus on the important frequency bands from the point of view of further data exploration. </li>
    <li> <b> 03_extract_features.ipynb </b> - This script is focused on data extraction, that is, the process of selecting and properly preparing features. This is an extremely important stage that directly affects the final results of the analysis. The most important extracted features are plots generated from the audio files: spectrograms, mel-spectrograms, chromagrams, MFCCs. Each graph was saved as a grayscale image in PNG format and as a NumPy array in NPY format. For MFCCs, 20 mel-cepstral coefficients were selected. In addition, a first mel-cepstral coefficients was removed from the MFCCs and the chart thus created was saved similarly to the rest of the attributes. For the mel-spectrogram, 128 mel bands were selected. </li>
    <li> <b> 04_cluster_audio.ipynb </b> - This script prepares the data for clustering and performs the clustering using classical clustering algorithms. To prepare the data, the code transforms it to one dimension and computes the corresponding means, checks if there are any missing values and replaces them with the means, normalizes the data, applies the PCA algorithm and estimates the number of clusters. Four algorithms (K-Means, Fuzzy C-Means, hierarchical agglomerative clustering and DBSCAN) were used for clustering. At first, the script visualizes the data with the original labels, searches for the best hyperparameters, performs clustering and presents the results in the form of visulization and tables. </li>
    <li> <b> 05_autoencoder.ipynb </b> - This script prepares the data to be used as input for the autoencoder, implements the autoencoder architecture, uses it to group the data together with the K-Means algorithm, and visualizes the results in the form of tables and images. </li>
    <li> <b> 00_prepare_US8K.ipynb </b> - A special script created for the UrbanSound8K dataset to transform its directory and file structure. </li>
    <li> <b> FeaturesDataset.py </b> - The python file that is needed to make 05_autoencoder.ipynb work correctly with the input data for the autonecoder </li>
</ul>

## Autoencoder
A convolutional autoencoder and the K-Means algorithm were used to group audio files. The artificial intelligence model was designed to generate a compressed form of the input data (immersion), and the K-Means algorithm was used to group such a representation of the data. The autoencoder architecture was taken from the paper [Deep Clustering with Convolutional Autoencoders][DCEC], in which a convolutional autoencoder is part of a larger Deep Convolutional Embedded Clustering (DCEC) neural network. 

![AE](/images-readme/ae.png)
*Simple autoencoder architecture*

The input data for the neural network were pre-generated images (spectrograms, mel-spectrograms, chromagrams, MFCCs, MFCCs without a zero line) with a size of 96x48. Such dimensions are due to two reasons: during experiments, the architecture received worse results with much larger images (i) and training models with larger images took too much time on local hardware (ii). 

![AE](/images-readme/cae.png)
*Implemented autonecoder architecture*

In order to test whether the autoencoder implementation is correct, the popular MNIST dataset was clustered. This made it possible to compare the results of the implemented autoencoder in this work with the results of other autoencoders.

![AE](/images-readme/mnist_reconstruction.png)
*MNIST reconstruction*

## Results
Tables with the best results for the number of clusters equal to the predefined number of labels. Much more analysis, visualizations and conclusions can be found in my master's thesis.

#### [Music Audio Benchmark Data Set][MABDS]
##### Classical Algorithms
| **Feature**                   | **Type**  | **Algorithm** | **NMI** | **ACC** |
|-------------------------------|-----------|---------------|---------|---------|
| Spectrogram                   | Images    | K-Means       | 14%     | 26%     |
| Mel-spectrogram                | Images    | FCM           | 12%     | 21%     |
| MFCCs                         | Arrays    | FCM           | 12%     | 28%     |
| MFCCs without first coefficient | Arrays    | FCM           | 11%     | 26%     |
| Chromagram                    | Arrays    | K-Means       | 6%      | 18%     |

##### Autoencoder
| **Feature**                   | **Embedding** | **NMI** | **ACC** |
|-------------------------------|---------------|---------|---------|
| Spectrogram                   | 1024          | 16%     | 29%     |
| Mel-spectrogram                | 128           | 15%     | 25%     |
| MFCCs                         | 128           | 8%      | 19%     |
| MFCCs without first coefficient | 1024          | 12%     | 21%     |
| Chromagram                    | 128           | 8%      | 21%     |

#### [UrbanSound8K][US8K]
##### Classical Algorithms
| **Feature**                   | **Type**  | **Algorithm** | **NMI** | **ACC** |
|-------------------------------|-----------|---------------|---------|---------|
| Spectrogram                   | Images    | K-Means       | 19%     | 25%     |
| Mel-spectrogram                | Arrays    | FCM      | 18%     | 24%     |
| MFCCs                         | Arrays    | K-Means  | 22%     | 31%     |
| MFCCs without first coefficient | Arrays    | K-Means  | 23%     | 32%     |
| Chromagram                    | Arrays    | K-Means       | 15%      | 21%     |

##### Autoencoder
| **Feature**                   | **Embedding** | **NMI** | **ACC** |
|-------------------------------|---------------|---------|---------|
| Spectrogram                   | 1024          | 20%     | 24%     |
| Mel-spectrogram                | 128           | 21%     | 26%     |
| MFCCs                         | 128           | 16%      | 22%     |
| MFCCs without first coefficient | 1024          | 17%     | 23%     |
| Chromagram                    | 128           | 17%      | 23%     |

#### [Toronto Emotional Speech Set][TESS]
##### Classical Algorithms
| **Feature**                   | **Type**  | **Algorithm** | **NMI** | **ACC** |
|-------------------------------|-----------|---------------|---------|---------|
| Spectrogram                   | Images    | Agglomerative| 86%     | 80%     |
| Mel-spectrogram                | Arrays    | Agglomerative| 78%     | 70%     |
| MFCCs                         | Arrays    | Agglomerative| 94%     | 89%     |
| MFCCs without first coefficient | Arrays    | Agglomerative| 93%     | 88%     |
| Chromagram                    | Images    | Agglomerative| 53% | 44%     |

##### Autoencoder
| **Feature**                   | **Embedding** | **NMI** | **ACC** |
|-------------------------------|---------------|---------|---------|
| Spectrogram                   | 128          | 96%     | 90%     |
| Mel-spectrogram                | 128           | 95%     | 88%     |
| MFCCs                         | 1024           | 63%      | 61%     |
| MFCCs without first coefficient | 128          | 79%     | 74%     |
| Chromagram                    | 128           | 91%      | 21%     |

### Conclusion based on thesis
The study analyzed audio file clustering methods, focusing on their effectiveness and the results obtained. In order to achieve the intended objectives, a literature review of existing research on audio file clustering was conducted, which enabled the selection of appropriate approaches and tools. Subsequently, three diverse datasets were selected, which in subsequent stages were processed and properly prepared for further analysis. Data clustering was carried out using classical algorithms such as DBSCAN, hierarchical agglomerative clustering, K-Means and Fuzzy C-Means, using available python libraries. In addition, a deep convolutional autoencoder was implemented, which was used to group the data using modern artificial intelligence techniques. The results of the experiments were visualized to better illustrate and analyze the results.

Clustering the three diverse datasets using different methods and evaluating them on several measures provided interesting insights. Although the results collected do not allow for clear conclusions, they provide a basis for drawing some interesting observations.

For classical algorithms, DBSCAN was found to be unreliable and usually achieved the worst results. Hierarchical agglomerative clustering showed potential, as it sometimes achieved the best results of all the algorithms tested, but it is often problematic due to its tendency to create large clusters and many small clusters that contained individual objects. Nevertheless, this algorithm is interesting because it allows the user to determine where to divide the data into clusters, so the clusters created during clustering can be controlled to some extent. K-Means and FCM performed very similarly and usually achieved good results. MFCCs and MFCCs without first coeffiecient saved as arrays, usually achieved the best results. The use of PCA had minimal impact on the results.

The results obtained with the simple autoencoder, which was originally designed to group small images from the MNIST dataset, proved comparable to classical algorithms for more complex audio data. This indicates the great potential of autoencoders and encourages further development and application of this clustering method in future studies. Smaller embeddings performed better in clustering, although they were slightly worse at reconstructing the original images than embeddings of larger size. The key, therefore, is to properly balance the size of the embeddings to ensure satisfactory reconstruction of the input data while keeping the size as small as possible, which promotes better clustering results. Among the features used, spectrograms and mel-spectrograms performed by far the best.

It is worth noting that the data analysis mainly relied on NMI and ACC measures, which have proven to be more reliable in assessing clustering quality than internal measures. These measures, while useful in assessing cluster compactness and separation, do not always correlate with actual clustering quality. Therefore, they should not be the sole criterion for evaluation.

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

[DCEC]: https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
