#-------------------------------------------------------------------------
# AUTHOR: Julia Chaidez
# FILENAME: decision_tree_2
# SPECIFICATION: train, test, and output the performance of 3 models created by using each training set on the test set provided
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    age = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism = {'No': 1, 'Yes': 2}
    tear = {'Reduced': 1, 'Normal': 2}

    for row in dbTraining:
        X.append([
            age[row[0]],
            spectacle[row[1]],
            astigmatism[row[2]],
            tear[row[3]]
        ])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    recLenses = {'Yes': 1, 'No': 2}

    for row in dbTraining:
        Y.append(recLenses[row[4]])

    accuracies = [] #store the accuracies

    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       dbTest = []
       with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # Skipping the header
                    dbTest.append(row)

       correctPredictions = 0
       totalPredictions = len(dbTest)
       
       for data in dbTest:
        #Transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        testSample = [
            age[data[0]],
            spectacle[data[1]],
            astigmatism[data[2]],
            tear[data[3]]
        ]

        # Predict the class
        class_predicted = clf.predict([testSample])[0]

        #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        if class_predicted == recLenses[data[4]]:
            correctPredictions += 1

    #Find the average of this model during the 10 runs (training and test set)
    accuracy = correctPredictions / totalPredictions
    accuracies.append(accuracy)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Final accuracy when training on {ds}: {average_accuracy:.2f}")




