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
#Target sum

""" def two_sum(numbers, target):
    if target in numbers and 0 in numbers:
        return (numbers.index(target),numbers.index(0))
    else:
        for i in range(len(numbers)-1):
            for j in range(i+1,len(numbers)):
                pointer1=numbers[i]
                pointer2=numbers[j]
                if pointer1+pointer2==target:
                    return (i,j)
                else:
                    pass """

#Big countries
#https://leetcode.com/problems/big-countries/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    flt=(world["area"]>=3000000) | (world["population"]>=25000000)
    return world[flt][["name","population","area"]] """


#Low fat and recyclable products
#https://leetcode.com/problems/recyclable-and-low-fat-products/submissions/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    flt=(products["low_fats"]=='Y') & (products["recyclable"]=='Y')
    return products[flt][["product_id"]] """


#Customers who never order
#https://leetcode.com/problems/customers-who-never-order/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
""" 
import pandas as pd

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    new_df=customers[~customers['id'].isin(orders['customerId'])]
    new2_df=new_df[['name']].rename(columns={"name":'customers'})
    return new2_df

"""

#Article views
#https://leetcode.com/problems/article-views-i/solutions/3852944/pandas-my-sql-very-simple-with-approach-and-explanation/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def article_views(views: pd.DataFrame) -> pd.DataFrame:
    view_own_article=views[views["author_id"]==views["viewer_id"]]
    distinct_authors=view_own_article["author_id"].unique()
    distinct_authors=sorted(distinct_authors)
    f_df=pd.DataFrame({"id":distinct_authors})
    return f_df """

#Invalid tweets
#https://leetcode.com/problems/invalid-tweets/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    flt=tweets["content"].str.len()>15
    return tweets[flt][["tweet_id"]] """