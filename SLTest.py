#MD Muhaimin Rahman 16/05/17
# I have edited the code for My project Bangladeshi Cricketers Recognition 
# The Real Code belongs to Siraj Raval , which is available in his repository
# It is your job to find out where I have edited the code and why . ;) 

import tensorflow as tf
import sys

# change this as you see fit
#image_path = sys.argv[1]

# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()
import os
import shutil
from os import listdir
from os import mkdir
from shutil import copyfile
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
varPath = '/home/sezan92/MEGA/SignLanguage/Test'
destDir = "/home/sezan92/MEGA/SignLanguage/TestTF"
imgFiles = [f for f in listdir(varPath) if isfile(join(varPath, f))]


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/home/sezan92/MEGA/SignLanguage/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/home/sezan92/MEGA/SignLanguage/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #try:
    #    shutil.rmtree(destDir)
    #except:
    #    None
    #mkdir ('scanned')
 
    for imageFile in imgFiles:
        image_data =  tf.gfile.FastGFile(varPath+"/"+imageFile, 'rb').read()       

        print (varPath+"/"+imageFile)
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        firstElt = top_k[0];

        newFileName = label_lines[firstElt] +"--"+ str(predictions[0][firstElt])[2:7]+".jpg"
        print(newFileName)
       # copyfile(varPath+"/"+imageFile, destDir+"/"+newFileName)
        scoreList = []
        human_stringList = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            scoreList.append(score)#print (node_id)
            human_stringList.append(human_string)
        plt.figure()
        testImage = cv2.imread(join(varPath,imageFile))
        plt.imshow(cv2.cvtColor(testImage,cv2.COLOR_BGR2RGB))
                
        plt.title('%s (score = %.5f)' % (human_stringList[scoreList.index(max(scoreList))
                  ], max(scoreList)))