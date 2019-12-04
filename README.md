## Team Members

|Name|Niner ID|Email ID|
|---|---|---|
|Dhananjay Arora| 801077164| darora2@uncc.edu|
|Arjun Kalidas| 801078014| akalidas@uncc.edu|
|Naman Manocha| 801077765| nmanocha@uncc.edu|

## Introduction
A recommendation system is a program/system that tries to make a prediction based on users’ past behavior and preferences. Recommendation systems are typically seen in applications such as music listening, watching movies and e-commerce applications where users’ behavior can be modeled based on the history of purchases or consumption. We see them all around and it benefits a user in many ways because of its nature of prediction, value-add, and ease of consumption. A user wouldn’t have to spend hours to decide about a particular movie or music or a product. The relevant and useful content gets delivered to the user at appropriate times. Our aim is to build such a recommendation system using the existing tools and technologies, in addition to adding our own flavor to the data parallelization aspect by using Spark and Deep Learning libraries such as Google’s TensorFlow, Keras or Scikit-learn [2]. We hope to achieve similar performance, if not better than the researchers that have been working on such technologies.

## Objective
The goal is to implement the SGD Stochastic Gradient Descent which helps in collaborative filtering so that we can suggest songs to the users.

## Tasks involved and approach

## Steps implemented

## Algorithms implemented

## Dataset
GroupLens[14] - HetRec 2011 - https://grouplens.org/datasets/hetrec-2011/
This dataset contains social networking, tagging, and music artist listening information from a set of 2K users from Last.fm online music system. http://www.last.fm 

Dataset files used are:
- artists.dat - This file contains information about music artists listened and tagged by the users.
- tags.dat - This file contains the set of tags available in the dataset.
- user_artists.dat - This file contains the artists listened by each user. It also provides a listening count for each [user, artist] pair.

This dataset was built by Ignacio Fernandez-Tobas with the collaboration of Ivan Cantador and Alejandro Bellogn, members of the Information Retrieval group at Universidad Autonoma de Madrid (http://ir.ii.uam.es)

Million Song Dataset [1] - http://static.echonest.com/millionsongsubset_full.tar.gz
Upon analysis of our dataset, we realized the magnitude was quite high to the ranks of millions. Since we cannot afford to run such a huge data and constrain the DSBA cluster or AWS, we are picking the smaller subset of 10,000 songs (1% of the entire dataset available from the source, 1.9GB). It contains "additional files" (SQLite databases) in the same format as those for the full set but referring only to the 10K song subset. Therefore, we can develop code on the subset, then port it to the full dataset.

## Context
We see an explosion of Music streaming apps these days and sometimes wonder or wrack our brains as to which one serves our purpose and how do we get the relevant set of songs when we open the application. We have many songs recommendation systems out there and those are being used by popular companies like Spotify, SoundCloud, Pandora, Amazon Music, etc. And most of them do predict music or movies based on our previous watching/listening history and feedback. Most of them work based on Collaborative and Content-based filtering which they call the “Hybrid” model [3]. The companies use advanced machine learning algorithms and data processing engines like Spark and Hadoop to produce the best results possible. While all these technologies exist, this is our take on the song recommendation systems and how we can contribute even though minuscule, to the already popular technology. We are aware that Machine learning algorithms like Neural Networks and Deep Learning are used in such complex systems. Along with these algorithms, we will leverage Alternating Least Squares in Spark [4] and execute the design on a distributed ecosystem like Hadoop, which is interesting.

## Final Result
The result is divided into 3 categories namely:
- What we will definitely accomplish

We will accomplish a music recommendation system based on Collaborative filtering and Contentbased filtering and achieve data parallelization using Spark.
- What we are likely to accomplish, and

Content-based and collaborative filtering as an input to a deep learning algorithm to improve recommendation accuracy and precision.
- What we would ideally like to accomplish.

An improved precision value above 88% on the already existing recommendation system based on deep learning. Also, we will attempt at overcoming the drawback of modeling based on negative recommendations due to insufficient data. And the way in which collaborative filtering data is obtained by calculating two values such as global information about every song and playlist, that is expensive to maintain, but this could be overcome by industry-standard techniques for indexing as well.

## Result and Examples

## Performance Evaluation (quantative)
- Interpretataion of results:
- Does the system work well
- Where it can be improved

## Task Division

|S.No. | TASK | PERSON | TIMELINE |
|---|---|---|---|
|1|Literature Survey|All|28-31 Oct|
|2|Problem Identification and Dataset Collection|All|28-31 Oct|
|3|Data Cleaning and Preprocessing|Dhananjay|7 Nov|
|4|Content and collaborative filtering|Naman and Arjun|16 Nov|
|5|Data Parallelization using Spark|Arjun and Dhananjay|21 Nov|
|6|Prediction using Deep Learning|Naman and Arjun|29 Nov|
|7|Experiments, Result Analysis, and Final Report|All|2 Dec|

We met every second day as we live together.

## Tools/Technologies/Frameworks used
- Apache Spark 2.4.4
- Python 3.7
- Java 1.8
- Jupyter Notebook 6.0.0
- UNCC-DSBA cluster/ AWS EMR cluster
- Git for hosting the website

## Packages used
- SciPy
- scikit-learn

## Challenges
- We had a tough time understanding the mathematical concepts behind content based and collaborative filtering.
- We were also confused at one point about latent features used in ALS algotithm.
- We were not sure how to translate our understanding of TF-IDF vectorization and cosine similarity that are applied on bag of words to work for songs playlist.
- Huge amount of data sets that we use for song recommendation let us into lot of performance related issues.
- Even the initial dataset we decided to implement this recommendation system had to be modified and additional datasets had to be merged to get all the required parameters.

## Things learnt
- We learnt to implement collaborative filtering using Stochastic Gradient Descent algorithm.
- Content based filtering using cosine similarity and TF-IDF vector.
- We learnt to pre-process and model the data.
- During our literature survey, we learnt different approaches to implement recommender system.

## Conclusion
- We accomplished a music recommendation system based on Collaborative filtering and Contentbased filtering and achieve data parallelization using Spark.

## References
[1] Paul Lamere, Million Song Dataset, Lab ROSA, Volume 137, 2011, http://millionsongdataset.com/pages/getting-dataset/

[2] F. Fessahaye et al., "T-RECSYS: A Novel Music Recommendation System Using Deep Learning,"2019 IEEE International Conference on Consumer Electronics (ICCE), Las Vegas, NV, USA, 2019, pp. 1-6. doi: 10.1109/ICCE.2019.8662028 https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8662028&isnumber=8661828

[3] Badr Ait Hammou, Ayoub Ait Lahcen, and Salma Mouline, “An effective distributed predictive model with Matrix factorization and random forest for Big Data recommendation systems”, Expert Systems with Applications, Volume 137, 2019, Pages 253-265, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2019.06.046

[4] Yazhong Feng, Yueting Zhuang, and Yunhe Pan, "Music information retrieval by detecting mood via computational media aesthetics," Proceedings IEEE/WIC International Conference on Web Intelligence(WI 2003), Halifax, NS, Canada, 2003, pp. 235-241. doi:10.1109/WI.2003.1241199, http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1241199&isnumber=27823

[5] Jack Vasylenko, Chitwan Kaudan, Anith Patel, Tyler Larsen, and William Wang, Spotify Million Playlists (RecSys 2018) Challenge Submission, https://github.com/vaslnk/Spotify-Song-Recommendation-ML

[6] Kevin Liao, Prototyping a Recommender System Step by Step Part 2: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering, https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1

[7] Albert Au Yeung, Matrix Factorization: A Simple Tutorial and Implementation in Python, http://www.albertauyeung.com/post/python-matrix-factorization/

[8] Alberta Yeung, Matrix Factorization, https://nbviewer.jupyter.org/github/albertauyeung/matrix-factorization-in-python/blob/master/mf.ipynb

[9] Insight, Explicit Matrix Factorization: ALS, SGD, and All That Jazz, https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

[10] Shuvayan Das, Beginners Guide to learn about Content Based Recommender Engines, https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/

[11] Ramzi Karam, Using Word2vec for Music Recommendations, Towards Data Science, https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484

[12] Paul Thompson, Matrix-Factorization-ALS, https://github.com/mickeykedia/Matrix-Factorization-ALS/blob/master/ALS%20Python%20Implementation.py

[13] Charles Bochet, Get Started with PySpark and Jupyter Notebook in 3 Minutes, Discover Sicara, https://www.sicara.ai/blog/2017-05-02-get-started-pyspark-jupyter-notebook-3-minutes

[14]  @inproceedings{Cantador:RecSys2011, author = {Cantador, Iv\'{a}n and Brusilovsky, Peter and Kuflik, Tsvi}, title = {2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)}, booktitle = {Proceedings of the 5th ACM conference on Recommender systems}, series = {RecSys 2011}, year = {2011}, location = {Chicago, IL, USA}, publisher = {ACM}, address = {New York, NY, USA}, keywords = {information heterogeneity, information integration, recommender systems},} 
