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