# -*- coding: utf-8 -*-
"""HW5
Lillie & Corinne
"""
import csv
import numpy as np
import math
import random

countryData = []

labels = []
countries = []
#K=5

def main():
    """ Runs program """
    readFile()
    [lowestCost, lowestCostCount, lowest_c, lowest_mu] = findLowestCost(countryData, 4, 100)
    makeClusters(lowest_c, lowest_mu)
    makeAverage(lowest_c)
    #testingKs(countryData, 4, 50)
    
def readFile():
    """ Open country data file and make list of lists of features per country
    Make labels list with each feature label
    """
    with open('country.csv') as countryFile:
        line_count = 0
        csv_reader = csv.reader(countryFile, delimiter= ',')
        for row in csv_reader:
            if line_count == 0:
                labels.append(row)
            else:
                intArray = []
                countries.append(row[0])
                for i in range(1, len(row)):
                    intArray.append(float(row[i]))
                array = np.array(intArray)
                countryData.append(array)
            line_count+=1

def distance(country1_feature_list, country2_feature_list):
    """ Computes the squared distance between two vectors"""
    sumNums = 0
    sum_squares = 0 
    for i in range(len(country1_feature_list)):
        sum_squares += (country1_feature_list[i] - country2_feature_list[i])**2
    sumNums += math.sqrt(sum_squares)
    return sumNums
    
def closestCenter(country_feature_list, centroids):
    """Takes in a data point (country_feature_list) and a list of centroids (centroids)
    and returns in the index of the closest cluster center for the data point using the 
    distance function """
    closestIndex = 0
    closestDistance = distance(country_feature_list, centroids[closestIndex])
    for i in range(len(centroids)):
        currentDistance = distance(country_feature_list, centroids[i])
        if currentDistance < closestDistance:
            closestIndex = i
            closestDistance = currentDistance
    return closestIndex

def clusterAssignment(countries, centroids):
    """Takes the list of data points (countries) and the list of centroids mu (centroids) and returns a list 
    c containing the indices of the closest cluster center for each data point."""
    c = []
    for i in range(len(countries)):
        closestIndex = closestCenter(countries[i], centroids)
        c.append(closestIndex)    
    return c

def moveCentroids(countryData, c,K):
    """This function takes the list of data points x, the list of cluster indices c, and K, and
    returns a list mu containing the K centroids based on the current assignment c."""
    mu = []
    for i in range(K):
        sumNums = np.zeros(16)
        numCountries = 0
        for j in range(len(c)):
            if c[j] == i:
                sumNums += np.array(countryData[j])
                numCountries += 1
        pointsAverage = np.true_divide(sumNums,numCountries)
        mu.append(pointsAverage)
    return mu

def calcCost(countryData, c, mu):
    """This function takes x, c, and mu, and computes the current cost J."""
    sum_distances = 0
    m = len(countryData);
    for i in range(m):
        sum_distances += (distance(countryData[i], mu[c[i]]))**2
    j = 1/m * sum_distances
    return j
    
def kMeans(x,K):
    """takes x and K and runs the K-means algorithm"""
    # initialize the cluster centers mu to the first K elements of x
    #mu = []
    #for i in range(K):
        #mu.append(x[i])
    mu = random.sample(x, K)
    
    c= []
    c_equal = False
    iterations = 0
    #cluster assignment step
    while (iterations < 500 and c_equal == False):
        new_c = clusterAssignment(countryData, mu)
        #print("cost after cluster assignment", calcCost(x, new_c, mu))
        
        count = 0
        for i in range(len(c)):
            if new_c[i] == c[i]:
                count+=1
            if(count == len(c)):
                c_equal = True
    
        if(c_equal == False):
            mu = moveCentroids(countryData, new_c,K)
            #print("cost after moving centroids", calcCost(x, new_c, mu))
        
        c = new_c
        iterations += 1
    
   # print("converged after:" + str(iterations-1))
         
    #print(c)
   
    return [mu, c] 
    

import matplotlib.pyplot as plt 


def findLowestCost(countryData, k, repititions):
    """Runs your randomized K-means function for a given number of repetitions,
    and keeps track of the minimum cost J obtained in each run, as well as how many times
    the lowest cost was found"""
    [mu, c] = kMeans(countryData, k)
    lowestCost = calcCost(countryData, c, mu)
    lowestCostCount = 1
    lowest_c = [] 
    
    for i in range(1, repititions):
        [mu,c] = kMeans(countryData, k)
        cost = calcCost(countryData, c, mu)
        if(cost < lowestCost):
            lowest_c = c
            lowest_mu = mu
            lowestCost = cost
            lowestCostCount = 1
        elif (cost == lowestCost):
            lowestCostCount += 1
    
    print("lowest c", lowest_c)    
    print("K: " + str(k))
    print("lowest cost: " + str(lowestCost))
    print("number of times found: " + str(lowestCostCount))
    return [lowestCost, lowestCostCount, lowest_c, lowest_mu]
    

def testingKs(countryData, K_range, reps):
    """ Runs algorithm for K = 1...30 and keeps track of the minimum costs J obtained. 
    Creates a plot of J vs. K using matplotlib """
    lowestCosts = []
    K_s = []
    
    for k in range(1, K_range+1):
        [lowestCost, lowestCostCount, c, mu] = findLowestCost(countryData, k, reps)
        lowestCosts.append(lowestCost)
        K_s.append(k)
    
    plt.plot(K_s, lowestCosts)
    plt.ylabel('J')
    plt.xlabel('K')

def makeClusters(c, mu):
    """ Iterates through c and makes list of country names for each cluster.
    It then calls addLabels to label each cluster center's values """

    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    
    for i in range(len(c)):
        if c[i] == 0:
            cluster1.append(countries[i])
        elif c[i] == 1:
            cluster2.append(countries[i])
        elif c[i] == 2:
            cluster3.append(countries[i])
        else:
            cluster4.append(countries[i])
    
    print("cluster1", cluster1)
    print(addLabels(mu[0]))
    print("cluster2", cluster2)
    print(addLabels(mu[1]))
    print("cluster3", cluster3)
    print(addLabels(mu[2]))
    print("cluster4", cluster4)
    print(addLabels(mu[3]))

def addLabels(mu):
    """ Uses list of labels to label each cluster center's values """
    labeledMu = {}
    for i in range(len(mu)):
       labeledMu[labels[0][i]] = mu[i]
    return labeledMu
        
    
def makeAverage(c):
    """ Finds average value of all priorities for all countries"""
    totalAverage = np.zeros(16)
   
    numCountries = 194
    
    for i in range(len(c)):
        totalAverage+= countryData[i]
    
   
    totalAverage = totalAverage/numCountries
    

    print("total average", addLabels(totalAverage))
    
if __name__ == "__main__":
    main()
   
