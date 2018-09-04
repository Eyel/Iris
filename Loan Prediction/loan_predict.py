import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

print "============================"
print "=   Exploratory Analysis   ="
print "============================\n"

print "FIRST 10 ROWS:\n"
print data.head(10)


# vue d'ensemble des donees, missing values
print "GLOBAL VIEW:\n"
print data.describe()

# pour categorical data
print "GLOBAL VIEW ON CATEGORICAL DATA:\n"
print "\n", 'Total', data['Gender'].count(),'\n', data['Gender'].value_counts()
print "\n", 'Total',data['Married'].count(),'\n',data['Married'].value_counts()
print "\n", 'Total',data['Education'].count(),'\n', data['Education'].value_counts()
print "\n", 'Total',data['Self_Employed'].count(),'\n',data['Self_Employed'].value_counts()
print "\n", 'Total',data['Property_Area'].count(),'\n', data['Property_Area'].value_counts()
print "\n", 'Total',data['Loan_Status'].count(),'\n',data['Loan_Status'].value_counts()

"""
#Distribution Income application
print "\nDISTRIBUTION -  ApplicantIncome :\n"
data['ApplicantIncome'].hist(bins=50)
plt.show()

#Distribution Loan application
print "DISTRIBUTION -  LoanAmount :\n"
data['LoanAmount'].hist(bins=50)
plt.show()

#Distribution loans application
print "BOX PLOT -  ApplicantIncome by marital status:\n"
data.boxplot(column='ApplicantIncome', by='Married')
plt.show()
print "BOX PLOT -  ApplicantIncome by education:\n"
data.boxplot(column='ApplicantIncome', by='Education')
plt.show()
print "BOX PLOT -  LoanAmount by Property_Area:\n"
data.boxplot(column='LoanAmount', by='Property_Area')
plt.show()
"""

# Missing Data
print "\nMISSING DATA\r"
print data.apply(lambda  x: sum(x.isnull()), axis=0)


#categorical Data

#Data scaling