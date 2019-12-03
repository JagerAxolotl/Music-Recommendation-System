# Course Project ITCS 6190 - Music Recommendation System

## Team Members:
Dhananjay Arora 801077164 darora2
Arjun Kalidas 801078014 akalidas
Naman Manocha 801077765 nmanocha

Topic: MUSIC RECOMMENDATION SYSTEM

Introduction:
A recommendation system is a program/system that tries to make a prediction based on users’ past behavior and preferences. Recommendation systems are typically seen in applications such as listening to music listening, watching movies and e-commerce applications where users’ behavior can be modeled based on the history of purchases or consumption. We see them all around and it benefits a user in many ways because of its nature of prediction, value-add, and ease of consumption. A user wouldn’t have to spend hours on end to think and decide on a particular movie or music or a product. The relevant and useful content gets delivered to the user at appropriate times. Our aim is to build such a recommendation system using the existing tools and technologies, in addition to adding our own flavor to the data parallelization aspect by using Spark and Deep Learning libraries such as Google’s Tensorflow, Keras or Scikit-learn. We hope to achieve similar performance, if not better than the researchers that have been working on such technologies.

Objective
The goal is to implement the SGD Stochastic Gradient Descent which helps in collaborative filtering. 



Dataset: 
Million Song Dataset - http://static.echonest.com/millionsongsubset_full.tar.gz
Dataset reference URL - http://millionsongdataset.com/pages/getting-dataset/

Upon analysis of our dataset, we realized the magnitude quite high to the ranks of millions. Since we cannot afford to run such a huge data and constrain the DSBA cluster or AWS, we are picking the smaller subset of 10,000 songs (1% of the entire dataset available from the source).

It contains "additional files" (SQLite databases) in the same format as those for the full set, but referring only to the 10K song subset. Therefore, we can develop code on the subset, then port it to the full dataset.
Additional Files
To get started they provide some additional files which are reverse indices of several types. These should come bundled with the core dataset.
List of all track Echo Nest ID. The format is: track id<SEP>song id<SEP>artist name<SEP>song title
(Careful, large to open in a web browser)
List of all artist ID. The format is: artist id<SEP>artist mbid<SEP>track id<SEP>artist name
The code to recreate that file is available here (and a faster version using the SQLite databases here).
List of all unique artist terms (Echo Nest tags).
List of all unique artist musicbrainz tags.
List of the 515.576 tracks for which we have the year information, ordered by year.
List of artists for which we know latitude and longitude.
Summary file of the whole dataset, meaning same HDF5 format as regular files, it contains all metadata but no arrays like audio analysis, similar artists and tags. Only 300 Mb.
SQLite database containing most metadata about each track (NEW VERSION 03/27/2011).
SQLite database linking artist ID to the tags (Echo Nest and musicbrainz ones).
SQLite database containing similarity among artists.

Context:
We see an explosion of Music streaming apps these days and sometimes wonder or wrack our brains as to which one serves our purpose and how do we get the relevant set of songs when we open the application. We have many songs recommendation systems out there and those are being used by conglomerates like Spotify, SoundCloud, Pandora, Amazon Music, etc. And most of them are able to predict our interests and movies based on our previous watching history and feedback. Most of them work based on Collaborative and Content-based filtering which they call the “Hybrid” model. The companies use advanced machine learning algorithms and data processing engines like Spark and Hadoop to produce the best results possible. While all these technologies exist, this is our take on the song recommendation systems and how we can contribute even though minuscule, to the already popular technology. We are aware that Machine learning algorithms like Neural Networks and Deep Learning are used in such complex systems. Along with those algorithms, we will leverage Alternating Least Squares in Spark and execute the design on a distributed ecosystem like Hadoop.

Final Result:
What you will definitely accomplish
We will definitely accomplish a music recommendation system based on Collaborative filtering and Content-based filtering - Hybrid approach. And we hope to achieve data parallelization using Spark
 What you are likely to accomplish, and 
Content-based and collaborative filtering as an input to a deep learning algorithm to improve recommendation accuracy and precision.
What you would ideally like to accomplish.
An improved precision value above 88% on the already existing recommendation system based on deep learning. Also, we will attempt at overcoming the drawback of negative recommendations due to insufficient data and expansive collaborative filtering because we employ Spark.

Task Division:

S.No.
TASK
PERSON
TIMELINE
1
Literature Survey



All
28-31 Oct
2
Problem Identification and Dataset Collection
All
28-31 Oct
3
Data Cleaning and Preprocessing
Dhananjay
7 Nov
4
Content and collaborative filtering  
Naman and Arjun
16 Nov
5
Data Parallelization using Spark
Arjun and Dhananjay
21 Nov
6
Prediction using Deep Learning 
Naman and Arjun
29 Nov
7
Experiments, Result Analysis, and Final Report 
All
2 Dec


Tools/Technologies used:
Apache Spark 2.4.4
Python 3.7
Java 1.8
Jupyter Notebook 6.0.0
UNCC-DSBA cluster/ AWS EMR cluster

References:
[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: http://dx.doi.org/10.1145/2827872

[2] F. Fessahaye et al., "T-RECSYS: A Novel Music Recommendation System Using Deep Learning," 2019 IEEE International Conference on Consumer Electronics (ICCE), Las Vegas, NV, USA, 2019, pp. 1-6. doi: 10.1109/ICCE.2019.8662028
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8662028&isnumber=8661828

[3] Badr Ait Hammou, Ayoub Ait Lahcen, Salma Mouline, “An effective distributed predictive model with Matrix factorization and random forest for Big Data recommendation systems”, Expert Systems with Applications, Volume 137, 2019, Pages 253-265, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2019.06.046

[4] Yazhong Feng, Yueting Zhuang, and Yunhe Pan, "Music information retrieval by detecting mood via computational media aesthetics," Proceedings IEEE/WIC International Conference on Web Intelligence (WI 2003), Halifax, NS, Canada, 2003, pp. 235-241. doi:10.1109/WI.2003.1241199
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1241199&isnumber=27823

[5] Paul Lamere, Million Song Dataset, Lab ROSA, Volume 137, 2011, http://millionsongdataset.com/pages/getting-dataset/
