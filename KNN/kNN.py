import argparse
import math
import csv
import numpy as np

##c1-------------------
parser = argparse.ArgumentParser(description='Input Parameters')
parser.add_argument('--data',  action = 'store', type = str, help = 'Please choose--Example.tsv or Gauss.tsv')
parser.add_argument("--k", help="no of clusters")

#Input arguments
args = parser.parse_args()
input_path = args.data
cluster= args.k
#c1-------------------------------
def weight_calculation(farthest,near,dist_i):
    # Calculating the weight as per weighting scheme
    if farthest == near:
        return 1
    else:
        weight = (farthest-dist_i)/(farthest-near)
        return weight

def classification(list_dist,list_obj):
    # Finding the class label as per kNN
    farthest = list_dist[len(list_dist)-1]
    near = list_dist[0]
    w_A = 0
    w_B = 0
    for l in range(len(list_dist)):
        if list_obj[l][0] == 'A':
            w_A += weight_calculation(farthest,near,list_dist[l])
        else:
            w_B += weight_calculation(farthest,near,list_dist[l])
    if w_A > w_B:
        return 'A'
    else:
        return 'B'

def distance(case_A,case_B):
    # Calculating the distance between two objects as per eucledian metric
    squared_diff = 0
    for a in range(1,len(case_A)):
        squared_diff += (case_A[a]-case_B[a])**2
    dist = math.sqrt(squared_diff)
    return dist

def kNN_Classifier(input_data,k,classify = False):
    case_base = [input_data[0]]
    list_obj_others = list()
    misclassification = 0
    # This loop is for forming the case_base
    for b in range(1,len(input_data)):
        dist = list()
        dictionary = {}
        list_obj = list()
        for c in range(len(case_base)):
            d = distance(input_data[b],case_base[c])
            dist.append(d)
            dictionary[d] = case_base[c]
        dist = sorted(dist)
        size = min(len(case_base),k)
        for e in range(size):
            list_obj.append(dictionary[dist[e]])
        label = classification(dist[0:size], list_obj)
        if label != input_data[b][0]:
            case_base.append(input_data[b])
        else:
            list_obj_others.append(input_data[b])

    for f in range(len(list_obj_others)):
        # This loop is for finding number of misclassification with respective to above case_base
        dist1 = list()
        dictionary1 = {}
        list_obj1 = list()
        for g in range(len(case_base)):
            d1 = distance(list_obj_others[f],case_base[g])
            dist1.append(d1)
            dictionary1[d1] = case_base[g]
        dist1 = sorted(dist1)
        for h in range(k):
            list_obj1.append(dictionary1[dist1[h]])
        label1 = classification(dist1[0:k], list_obj1)
        if label1 != list_obj_others[f][0]:
            misclassification += 1
    if classify:
        return misclassification
    else:
        return case_base

# Reading csv file
with open(input_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    input_data = list()
    actual_class = list()
    for row in tsvreader:
        x = [float(x) for x in row[1:3]]
        data = [row[0]]
        data.extend(x)
        input_data.append(data)
    k = cluster
    misclassification = list()
    misclassification.append(kNN_Classifier(input_data,k,True))
output_vector = kNN_Classifier(input_data,k,False)

with open('result.csv',"w+",newline='') as csv_file:
    print(','.join(str(d) for d in misclassification), file=csv_file)
    csv_writer = csv.writer(csv_file, delimiter=",")
    for row in output_vector:
        csv_writer.writerow(row)

