#Find the  employees that are earning more than their managers
#https://leetcode.com/problems/employees-earning-more-than-their-managers/description/?lang=pythondata

'''
def find_employees(df):
    l=[]
    for i in range(len(employee)):
        if employee.loc[i,"managerId"]==None:
            pass
        else:
            if int(employee.loc[i,"salary"])>int(employee.loc[int(employee.loc[i,"managerId"])-1,"salary"]):
                l.append(employee.loc[i,"name"])
        return l

'''
'''
import pandas as pd

def find_employees(employee: pd.DataFrame) -> pd.DataFrame:
    for i in range(len(employee)):
        if employee.loc[i,"managerId"]==None:
            pass
        else:
            if int(employee.loc[i,"salary"])>int(employee.loc[int(employee.loc[i,"managerId"])-1,"salary"]):
               return employee[employee["name"]==employee.loc[i,"name"]]
'''

#Return the difference between the min and the max element on a list

""" 
def max_diff(lst):
    if len(lst)<=1:
        return 0
    else:
        minimal=min(lst)
        maximal=max(lst)
        return maximal-minimal """

#Find repeated emails
#https://leetcode.com/problems/duplicate-emails/?lang=pythondata

""" import pandas as pd
from collections import Counter

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
    l=[]
    d=Counter(person["email"])
    for i,j in d.items():
        if j>1:
            l.append(i)
    for email in l:
        return person[person["email"]==email] """

#Creating a spiral box
#https://www.codewars.com/kata/63b84f54693cb10065687ae5/solutions

""" def create_box(m, n):
    # Create an empty 2D list with dimensions m x n
    matrix = [[0] * m for _ in range(n)]
    
    # Fill the first and last row with 1s
    for i in range(m):
        matrix[0][i] = 1
        matrix[n-1][i] = 1
        
    # Fill the first and last column with 1s
    for i in range(n):
        matrix[i][0] = 1
        matrix[i][m-1] = 1
        
    # Fill in the remaining values
    for i in range(1, n-1):
        for j in range(1, m-1):
            val = min(i, j, n-i-1, m-j-1) + 1
            matrix[i][j] = val
    
    return matrix """

""" #https://leetcode.com/problems/customers-who-never-order/?lang=pythondata
#Return customers that haven't order anything

import pandas as pd

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    customers.drop(index=list(orders["customerId"]))
    return customers["name"]

 """