#-------------------------------------------------------------------------
# AUTHOR: Julia Chaidez
# FILENAME: knn
# SPECIFICATION: Read the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Count misclassified instances
misclassified = 0
numSamples = len(db)

#Convert each feature value to float to avoid warning messages
for i in range(len(db)):
    db[i][:20] = [float(value) for value in db[i][:20]]

#Loop your data to allow each instance to be your test set
for i in range(numSamples):
    X = []
    Y = []

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration
    for j in range(numSamples):
        if j != i:
           X.append(db[j][:20]) # Attributes
           Y.append(db[j][20])  # Class label

    #Store the test sample
    testSample = db[i][:20]
    trueLabel = db[i][20]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction
    classPredicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate
    if classPredicted != trueLabel:
       misclassified += 1
    
errorRate = misclassified / numSamples

#Print the error rate
print("The LOO-CV error rate for a 1NN classifier on the spam/ham classification task is", errorRate)






