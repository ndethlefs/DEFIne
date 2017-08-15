libraries used:
- mpl_toolkits
- matplotlib
- numpy
- tkinter
- pandas
- sklearn
- pydotplus
- pylab
- kmodes
- os
- imageio
- pickle

Aim of code was to allow user to use classification and clustering techniques without the technical knowledge. Code is split into 3 pieces cmd_build_classifiers_clusters is used to build the classification and  clustering object which are then saved ready to be used in load_classifiers_and_clusters. This piece of code loads objects in and allow the user to give it more data. 

Tested against:
- iris (classification) : works
- abalone (classification) : works
- adult (classification) : works
- bank (classification) : works
- poker hand (classification) : works
- wine (classification) : works
- car (classification) : doesn’t work

ToDo:
- when saving classifers save a notepad which has the accuracy of the classifers against the test data set (1 data set spluit into training and testing)
- currently when finding k it does num of clusters from 2 – 10 this may need to be changed to something more realistic
- find out why doesn’t work with car dataset
- add unsupervised classification
- auto detect best k from the outputted elbow method (currently relies on user choosing the best k using this method)
- make a pip install of the code
