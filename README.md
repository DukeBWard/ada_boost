# ada_boost
Decision tree and adaboost

train \<examples> \<hypothesisOut> <learning-type> should read in labeled examples and perform some sort of training.
examples is a file containing labeled examples. For example.
hypothesisOut specifies the file name to write your model to.
learning-type specifies the type of learning algorithm you will run, it is either "dt" or "ada". 

predict \<hypothesis> \<file> 
hypothesis is a trained decision tree or ensemble created by your train program
file is a file containing lines of 15 word sentence fragments in either English or Dutch. For example.
