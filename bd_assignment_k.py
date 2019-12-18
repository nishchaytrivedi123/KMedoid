import csv
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

#input data file
data = pd.read_csv('data.csv')

#dissimilarity matrix
dis_mattrix = squareform(pdist(data))

#input from use. can take any integer value
k = input("Enter value of k: ")

k = int(k)

#randomly chosen k samplesas centroid
samples = data.sample(n=k)
centroid = []
clusters = {}

#based on centroids they are assigned to clusters
for l in range(len(samples)):
	clusters[l] = [samples.iloc[l].name]
	centroid.append(samples.iloc[l].name)

# this is the main function that looks for distance and based on that make instance fall in according clusters
def main(centroid, clusters):
	for j, val in data.iterrows():
		key = 0
		dist, min_dist = 0, 1000
		for i in range(len(centroid)):
			if i!= j:
				dist = dis_mattrix[centroid[i],j]
				if dist < min_dist:
					min_dist = dist
					key = i

		clusters[key].append(j)
	return clusters

clusters = main(centroid, clusters)

# this loop iterate for 100 iteration to check whether clusters do not change
for iterr in range(100):
	old_centroid = centroid
	centroid = []

	# after every iteration new centroid is taken based on shortest distance from that point to other in same cluster
	for key, val in clusters.items():
		cluster_dist = {}
		for i in clusters[key]:
			calc_distance = 0
			for j in clusters[key]:
				calc_distance += dis_mattrix[i, j]
			cluster_dist[i] = calc_distance
		
		centroid.append(min(cluster_dist.keys(), key=(lambda k: cluster_dist[k])))
	

	for l in range(len(centroid)):
		clusters[l] = [centroid[l]]

	# function is called to perform calculation
	clusters = main(centroid, clusters)

	#compared with the previous one if comes same then co points differ from cluster and we have our final clustering solution
	# so we wiil break loop and do not iterate all 100 iterations
	if old_centroid == centroid:
		break

# this snippet is to calculate average Silhouette Width
s_avg_clust = []
for key, val in clusters.items():
	s_sum = 0
	for v in val:
		a_sum = 0
		for w in val:
			a_sum += dis_mattrix[v, w]
		# calculate the average distance from same cluster and that average distance will be assigned to aa
		a = a_sum/len(val)
		b_sum_list = []
		for i in range(k):
			b_sum = 0
			if key != i:
				for j in clusters[i]:
					b_sum += dis_mattrix[j, v]

				# from a point to average distance from rest of the clusters in which it does not fall is calculated
				b_sum_list.append(b_sum/len(clusters[i])) 
		# minimum of calculated distance is taken
		b = min(b_sum_list)
		# Silhouette width of that point is calclulated
		s = (b-a)/max(a,b)
		s_sum += s

	s_avg_clust.append(s_sum/len(clusters[key])) # average Silhouette width of each cluster is calculated
print("Average Silhouette Width: ", sum(s_avg_clust)/len(s_avg_clust)) # final average Silhouette width of final clustering solution


# to make a new column to dataframe as cluster_number
new_dict = {}
for key, val in clusters.items():
	for v in val:
		new_dict[v] = key+1

data['cluster_number'] = new_dict.values()

# write to a file cluster_k based on k
data.to_csv('cluster_'+str(k)+'.csv', index=False)