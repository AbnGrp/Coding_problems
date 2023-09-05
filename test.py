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

#Track the queen
#https://www.codewars.com/kata/5aa1031a7c7a532be30000e5/train/python
