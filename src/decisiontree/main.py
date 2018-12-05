from decisiontree.DecisionTree import *

from images import featureextraction as FE
from images import *

fe = FE.make_extractor()
#import samples, directory can change as needed
training_samples = Sample.get_samples("C:/Users/osage/Desktop/SL Short Project/ISIC-images/UDA-1")
examples = [] #fill with [X1,X2,...,Xn, Y]

for sample in training_samples:
    #create grey-image
    grey_image = fe.convert_image_to_greyscale(sample)
    features = fe.get_features(grey_image)
    obs = 1 if sample.diagnosis == "melanoma" else 0
    features.append(obs)

    examples.append()

tree_depth_5 = growTree(examples)
tree_5_prediction = tree_depth_5.predict(examples)




#tree_depth_10 = growTree(training_features,10)
#tree_depth_10.predict(training_features)

#tree_depth_20 = growTree(training_features,20)
#tree_depth_20.predict(training_features)

#tree_depth_50 = growTree(training_features,50)
#tree_depth_50.predict(training_features)

