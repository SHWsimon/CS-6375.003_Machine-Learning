#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:32:00 2018

@author: simonwang
"""
import re;
import sys;
import json;

class KMeans:
    def __init__(self, K, tweets, seeds, output):
        self.K = K;
        self.tweets = tweets; #Tweets.json: id, text, word
        self.seeds = seeds;
        self.output = output;
        self.centroid = {}; #centroid of each cluster
        
        self.clusters = {}; #clustered by seed's tweetID 
        self.rev_clusters = {}; #clustered by tweetID to index
        self.jaccard_matrix = {}; #matrix stores pairwise jaccard distance
        
        self.new_clusters = {}; #clustered by tweetID 
        self.new_rev_cluster = {}; #clustered by tweetID to index
        
        self.initialClusters(); #initial clusters and rev_clusters
        self.initialMatrix(); #calculate jaccard_matrix 
               
    def initialClusters(self):
        #Tweets to no cluster
        for ID in self.tweets:
            self.rev_clusters[ID.id] = -1;

        #Clusters with seeds
        for k in range(self.K):
            self.clusters[k] = set([self.seeds[k]]);
            self.rev_clusters[self.seeds[k]] = k; 
            
    def jaccardDistance(self, ID1, ID2):
        SetA = set(ID1.word);
        SetB = set(ID2.word);
        
        return 1 - float(len(SetA.intersection(SetB))) / float(len(SetA.union(SetB)));
     
    def initialMatrix(self):
        for ID1 in self.tweets:
            self.jaccard_matrix[ID1.id] = {};
            for ID2 in self.tweets:
                if ID2.id not in self.jaccard_matrix:
                    self.jaccard_matrix[ID2.id] = {};
                distance = self.jaccardDistance(ID1, ID2);
                self.jaccard_matrix[ID1.id][ID2.id] = distance;
                self.jaccard_matrix[ID2.id][ID1.id] = distance;
 
    #Build new cluster
    def newClusters(self):
        for k in range(self.K):
            self.new_clusters[k] = set();

        for ID1 in self.tweets: 
            min_dist = sys.float_info.max;
            min_cluster = self.rev_clusters[ID1.id];

            #Calculate distance to each cluster
            for k in self.clusters:
                dist = 0.0;
                seedsNum = 0;
                for ID2 in self.clusters[k]:
                    dist += self.jaccard_matrix[ID1.id][int(ID2)];
                    seedsNum += 1;
                if seedsNum > 0:
                    avg_dist = dist / float(seedsNum);
                    if min_dist > avg_dist:
                        min_dist = avg_dist;
                        min_cluster = k;
            self.new_clusters[min_cluster].add(ID1.id);
            self.new_rev_cluster[ID1.id] = min_cluster;
        
        #Count center
        for k in self.new_clusters:
            num_id = len(self.new_clusters[k]);
            if num_id != 0:
                min_dist = sys.float_info.max;
                centroid_temp = self.new_clusters[k];

                for ID1 in self.new_clusters[k]: #temp_center
                    dist_sum = 0.0;
                    for ID2 in self.new_clusters[k]: #rest point
                        if ID1 != ID2:
                            dist_sum += self.jaccard_matrix[ID1][ID2];
                    if dist_sum < min_dist:
                        min_dist = dist_sum;
                        centroid_temp = ID1;
                self.centroid[k] = centroid_temp; 
                
        #Write the output to profile  
        report = open(self.output, 'w');
        self.writeOutput(report);
        
        #SSE
        sse = self.calculateSSE();
        report.write("\nSSE : %f\n"  % sse);
        
        report.close(); 
    
    def writeOutput(self, resultsoutput=None):
        clusterNum = 1;
        clusters_id=[];
        for k in range(self.K):
            for cluster in self.new_clusters[k]:
                clusters_id.append(cluster);
            resultsoutput.write( "%d\n%s\n" % (clusterNum, clusters_id) );
            clusters_id=[];
            clusterNum += 1
      
    def calculateSSE(self):
        SSE = 0.0
        for k in range(self.K):
            for ID in self.new_clusters[k]: #all ID in cluster k
                SSE += (self.jaccard_matrix[self.centroid[k]][ID])**2
        return SSE
      
#Preprocessing tweet data           
class tweetSplit:
    def __init__(self, tweet_id, tweet_text):
        self.id = tweet_id; #tweet id
        self.text = tweet_text; #tweet text
        self.word = self.text.split(); #each word of tweet
    def __repr__(self):
        return str(self.id) + ":" + self.text;

def readTweet(file):
    tweets_idText = [];
    for line in open(file, 'r'):
        tweetInfo = json.loads(line);
        # remove URL, retweet, hashtag, Unicode
        tweetInfo["text"] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweetInfo["text"]);
        tweetInfo["text"] = re.sub('^(RT @.+?: )+', '', tweetInfo["text"]);
        tweetInfo["text"] = re.sub('#\w+', '', tweetInfo["text"]);
        tweetInfo["text"] = re.sub("[^\w ]", ' ', tweetInfo["text"]);
        
        #store id and text 
        dataPre = tweetSplit(tweetInfo["id"], tweetInfo["text"]);
        tweets_idText.append(dataPre);
    
    return tweets_idText; 


if __name__ == "__main__":
    #Input
    K = int(sys.argv[1]);
    initSeeds = sys.argv[2];
    tweetRawData = sys.argv[3];
    output = sys.argv[4];

    #Preprocessing
    tweetRawData_pre = readTweet(tweetRawData); 
    
    #Seeds
    seeds = [];
    f = open(initSeeds, 'r');
    for ID in f.readlines():
        ID = re.sub('[\s,]+', '', ID);
        seeds.append(ID); 
    f.close(); 
    
    #Clustering
    kmeans = KMeans(K, tweetRawData_pre, seeds, output);
    kmeans.newClusters();
    