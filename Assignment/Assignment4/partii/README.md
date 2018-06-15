Language Used: Python 3.6

#Tweets-Clustering-KMeans
Implement the tweet clustering function using the Jaccard Distance metric and K-means clustering algorithm to cluster tweets into the same cluster. The algorithm is wirtten in python.

#Files included:
1. InitialSeeds.txt-contains the initial centroids of the k-means
2. tweets_Clustering.py - the python script for tweets clustering using k-means
3. Tweets.json - the boston bombing tweets dataset
4. tweets-k-means-output.txt - output file

#Packages
re - process string
sys - read the command line argument
json - read json file

#Step to run the code:
Excute the follwoing command to run the python program on the command line

    python3 tweetsClustering.py <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>

e.g.:

    python3 tweetsClustering.py 25 InitialSeeds.txt Tweets.json tweets-k-means-output.txt

#Output
1. tweets-k-means-output.txt
2. SSE = 9.696069


