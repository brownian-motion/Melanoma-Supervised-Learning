from decisiontree.DecisionTree import *


import featureextraction as FE
from images import *

fe = FE.make_extractor()
#import samples, directory can change as needed
training_samples = Sample.get_samples("C:/Users/osage/Desktop/SL Short Project/ISIC-images/UDA-1")
#for i in range(0,training_samples):
    #extact im array for all images to train the tree
    #training_im_array = [fe.get_image_arr(training_samples.get_grayscale_image()) for sample in training_samples]
    #training_feature_list.append([fe.get_features(training_im_array) for sample in training_samples]


#tree_depth_5 = growTree(training_features)
#tree_depth_5.predict(training_features)

#tree_depth_10 = growTree(training_features,10)
#tree_depth_10.predict(training_features)

#tree_depth_20 = growTree(training_features,20)
#tree_depth_20.predict(training_features)

#tree_depth_50 = growTree(training_features,50)
#tree_depth_50.predict(training_features)

