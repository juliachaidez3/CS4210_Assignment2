#-------------------------------------------------------------------------
# AUTHOR: Julia Chaidez
# FILENAME: naive_bayes
# SPECIFICATION: read the file weather_training.csv and output the classification of each of the 10 instances from the file weather_test
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []

#Reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity = {'High': 1,'Normal': 2}
wind = {'Weak': 1, 'Strong': 2}

X = []
for row in db:
   X.append([
      outlook[row[1]],
      temperature[row[2]],
      humidity[row[3]],
      wind[row[4]]
   ])

#Transform the original training classes to numbers and add them to the vector Y.
playTennis = {'Yes': 1,'No': 2}

Y = []
for row in db:
   Y.append(playTennis[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
   reader = csv.reader(csvfile)
   header = next(reader)  # Skip header

test_X = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        test_X.append([
            outlook[row[1]],
            temperature[row[2]],
            humidity[row[3]],
            wind[row[4]]
        ])

#Printing the header os the solution
print("Day    Outlook   Temperature Humidity Wind   PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
predictions = clf.predict(test_X)
probabilities = clf.predict_proba(test_X)

test_db = []
day_ID = []

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip header

    for row in reader:
        day_ID.append(row[0])
        test_X.append([
            outlook[row[1]],
            temperature[row[2]],
            humidity[row[3]],
            wind[row[4]]
        ])
        test_db.append(row)

for i, (prediction, probs) in enumerate(zip(predictions, probabilities)):
    confidence = max(probs)
    
    if confidence >= 0.75:
        label = "Yes" if prediction == 1 else "No"
        print(f"{day_ID[i]:<6} {test_db[i][1]:<9} {test_db[i][2]:<11} {test_db[i][3]:<8} {test_db[i][4]:<6} {label:<10} {confidence:.2f}")