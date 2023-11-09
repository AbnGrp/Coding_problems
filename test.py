#Pandas basics

#Get the size of a dataframe
#https://leetcode.com/problems/get-the-size-of-a-dataframe/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def getDataframeSize(players: pd.DataFrame) -> List[int]:
    return list(players.shape)
 """

#Display the first three rows
#https://leetcode.com/problems/display-the-first-three-rows/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.head(3) """

#Select data
#https://leetcode.com/problems/select-data/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def selectData(students: pd.DataFrame) -> pd.DataFrame:
    flt=students["student_id"]==101
    return students[flt][["name","age"]] """

#Create a new column
#https://leetcode.com/problems/create-a-new-column/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees["bonus"]=employees["salary"]*2
    return employees """

#Data cleaning
#https://leetcode.com/problems/drop-duplicate-rows/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.drop_duplicates(subset=["email"],keep="first") """

#Drop missing data
#https://leetcode.com/problems/drop-missing-data/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata
""" 
import pandas as pd

def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
    return students.dropna() """

#Modify columns
#https://leetcode.com/problems/modify-columns/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def modifySalaryColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees["salary"]=employees["salary"]*2
    return employees """

#Rename columns
#https://leetcode.com/problems/rename-columns/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
    students.rename(columns={"id":"student_id","first":"first_name","last":"last_name","age":"age_in_years"},inplace=True)
    return students """

#Change datatype
#https://leetcode.com/problems/change-data-type/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    students["grade"]=students["grade"].astype("int")
    return students """

#Fill missing data
#https://leetcode.com/problems/fill-missing-data/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products["quantity"]=products["quantity"].fillna(0)
    return products """

#Reshape the data
#https://leetcode.com/problems/reshape-data-concatenate/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    vertical_df=pd.concat([df1,df2],axis=0)
    return vertical_df """

#Pivot
#https://leetcode.com/problems/reshape-data-pivot/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def pivotTable(weather: pd.DataFrame) -> pd.DataFrame:
    return weather.pivot(index='month', columns='city', values='temperature') """

#Melt
#https://leetcode.com/problems/reshape-data-melt/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata

""" import pandas as pd

def meltTable(report: pd.DataFrame) -> pd.DataFrame:
    return pd.melt(report, id_vars = 'product', var_name = "quarter", value_name = "sales") """

#Method chaining
#https://leetcode.com/problems/method-chaining/description/?envType=study-plan-v2&envId=introduction-to-pandas&lang=pythondata
""" 
import pandas as pd

def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
    animals=animals.sort_values(by=["weight"],ascending=False)
    flt=animals["weight"]>100
    return animals[flt][["name"]] """

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

#Calculate bonus
#https://leetcode.com/problems/calculate-special-bonus/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus'] = 0
    
    employees.loc[(employees['employee_id'] % 2 != 0) & (~employees['name'].str.startswith('M')), 'bonus'] = employees['salary']
    result_df = employees[['employee_id', 'bonus']].sort_values(by='employee_id', ascending=True)
    
    return result_df """

#Fix names in a table
#https://leetcode.com/problems/fix-names-in-a-table/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
""" 
import pandas as pd

def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    users["name"]=users["name"].str.capitalize()
    return users.sort_values(by=["user_id"]) """

#Invalid emails
#https://leetcode.com/problems/find-users-with-valid-e-mails/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    # Use the str.match() method with a regex pattern to find valid emails
    valid_emails_df = users[users['mail'].str.match(r'^[A-Za-z][A-Za-z0-9_\.\-]*@leetcode(\?com)?\.com$')]
    
    return valid_emails_df """

#Pattients with a condition
#https://leetcode.com/problems/patients-with-a-condition/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    flt=patients["conditions"].str.contains(r'\bDIAB1')
    return patients[flt] """

#Delete duplicate emails
#https://leetcode.com/problems/delete-duplicate-emails/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

# Modify Person in place
def delete_duplicate_emails(person: pd.DataFrame) -> None:
  person.sort_values(by=["id"],ascending=True,inplace=True)
  person.drop_duplicates(subset=["email"],keep="first",inplace=True) """

#Rearrange products table
#https://leetcode.com/problems/rearrange-products-table/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:
    return pd.melt(
        products, id_vars='product_id', var_name='store', value_name='price'
    ).dropna() """

#Total time spent by each employee
#https://leetcode.com/problems/find-total-time-spent-by-each-employee/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    employees['total_time'] = employees['out_time'] - employees['in_time']

    empl = employees.groupby(by=['emp_id','event_day'],as_index=False).sum()

    return empl[['event_day','emp_id','total_time']].rename(columns={'event_day':'day'}) """

#Game play Analysis I
#https://leetcode.com/problems/game-play-analysis-i/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
     # Sort the DataFrame by player_id and event_date
    activity = activity.sort_values(by=['player_id', 'event_date'])
    
    # Group by player_id and select the minimum event_date for each player
    result = activity.groupby('player_id')['event_date'].min().reset_index()
    result.rename(columns={'event_date': 'first_login'}, inplace=True)
    
    return result """

#Number of unique subjects taught by each teacher
#https://leetcode.com/problems/number-of-unique-subjects-taught-by-each-teacher/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

result=teacher.groupby("teacher_id")["subject_id"].nunique().reset_index()
result.rename(columns={"subject_id":"cnt"})
return result """

#Classes with more than five students
#https://leetcode.com/problems/classes-more-than-5-students/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    class_count=courses.groupby("class")["student"].count().reset_index()
    result=class_count[class_count["student"]>=5][["class"]]
    return result """

#Customer placing the largest number of orders
#https://leetcode.com/problems/customer-placing-the-largest-number-of-orders/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    order_count=orders.groupby("customer_number")["order_number"].count().reset_index()
    val=max(order_count["order_number"],default=0)
    return order_count[order_count["order_number"]==val][["customer_number"]] """

#Group sold products by the date
#https://leetcode.com/problems/group-sold-products-by-the-date/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    return activities.groupby('sell_date')['product'].agg([('num_sold','nunique'),('products',lambda x: ','.join(sorted(x.unique())))]).reset_index()
 """

#Daily Leads and Partners
#https://leetcode.com/problems/daily-leads-and-partners/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

""" import pandas as pd

def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    new_c=daily_sales.groupby(["date_id","make_name"]).agg({"lead_id":"nunique","partner_id":"nunique"}).reset_index()
    new_c.columns=["date_id","make_name","unique_leads","unique_partners"]
    return new_c """

#Actors and directors
#https://leetcode.com/problems/actors-and-directors-who-cooperated-at-least-three-times/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
""" 
import pandas as pd

def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    new_df=actor_director.groupby(["actor_id","director_id"])["timestamp"].count().reset_index()
    return new_df[new_df["timestamp"]>=3][["actor_id","director_id"]] """

#Replace employee ID
#https://leetcode.com/problems/replace-employee-id-with-the-unique-identifier/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
""" 
import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    new_df=employees.merge(employee_uni,how="left",on="id")
    return new_df[["unique_id","name"]] """

""" return pd.merge(
        left=pd.merge(
            students, subjects, how='cross',
        ).sort_values(
            by=['student_id', 'subject_name']
        ),
        right=examinations.groupby(
            ['student_id', 'subject_name'],
        ).agg(
            attended_exams=('subject_name', 'count')
        ).reset_index(),
        how='left',
        on=['student_id', 'subject_name'],
    ).fillna(0)[
        ['student_id', 'student_name', 'subject_name', 'attended_exams']
    ] """

#Sales person
#https://leetcode.com/problems/sales-person/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata

#Partial solution
""" 
import pandas as pd

def sales_person(sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    new_df=orders.merge(sales_person,how="inner",on="sales_id")
    return sales_person[~(sales_person["name"].isin(new_df[new_df["com_id"]==1]["name"]))][["name"]] """

#Solution

""" import pandas as pd

def sales_person(sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    intermediate_merge = sales_person.merge(right=orders, how="outer", on="sales_id")

    final_merge = intermediate_merge.merge(right=company, how="left", on="com_id")

    red = final_merge[final_merge["name_y"] == "RED"][["name_x"]]

    non_red = sales_person[["name"]].merge(right=red, how="outer", left_on="name", right_on="name_x", indicator=True)

    return non_red[non_red["_merge"] == "left_only"][["name"]] """

#Combine two tables
#https://leetcode.com/problems/combine-two-tables/description/?lang=pythondata

""" import pandas as pd

def combine_two_tables(person: pd.DataFrame, address: pd.DataFrame) -> pd.DataFrame:
    new_df=person.merge(address,how="left",on="personId")
    return new_df[["firstName","lastName","city","state"]] """

#Duplicate emails
#https://leetcode.com/problems/duplicate-emails/description/?lang=pythondata

""" import pandas as pd

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
   results = pd.DataFrame()

   results = person.loc[person.duplicated(subset=['email']), ['email']]
    
   return results.drop_duplicates() """

#Triangle judgement
#https://leetcode.com/problems/triangle-judgement/

""" import pandas as pd
import numpy as np

def triangle_judgement(triangle: pd.DataFrame) -> pd.DataFrame:
    x = triangle["x"].values
    y = triangle["y"].values
    z = triangle["z"].values

    conditions = np.logical_or(np.logical_or(x + y <= z, x + z <= y), y + z <= x)
    triangle["triangle"] = np.where(conditions, "No", "Yes")

    return triangle """

#Top travelers
#https://leetcode.com/problems/top-travellers/
""" 
import pandas as pd

def top_travellers(users: pd.DataFrame, rides: pd.DataFrame) -> pd.DataFrame:
    rides=rides.groupby("user_id")["distance"].sum().reset_index()
    new_df=users.merge(rides,how="left",left_on="id",right_on="user_id")
    new_df.rename(columns={"distance":"travelled_distance"},inplace=True)
    new_df.sort_values(by=["travelled_distance","name"],ascending=[False,True],inplace=True)
    return new_df.fillna(0)[["name","travelled_distance"]] """

#List the products ordered in a period
#https://leetcode.com/problems/list-the-products-ordered-in-a-period/description/

""" import pandas as pd

def list_products(products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:

    new_orders=orders[(orders["order_date"].dt.month==2)&(orders["order_date"].dt.year==2020)]
    new_orders.drop(columns=["order_date"],inplace=True)
    new_df=new_orders.groupby("product_id")["unit"].sum().reset_index()
    f_df=products.merge(new_df,how="inner",on="product_id")
    return f_df[f_df["unit"]>=100][["product_name","unit"]] """

#Number of employes under the same manager
#https://leetcode.com/problems/the-number-of-employees-which-report-to-each-employee/

""" import pandas as pd
def list_products(employees: pd.DataFrame) -> pd.DataFrame:
 
    new_df=employees.merge(employees,how="inner",left_on="employee_id",right_on="reports_to")
    f_df=new_df.groupby(["employee_id","name"]).agg({"reports_to_x":"count","age_x":"mean"})
    f_df.rename(columns={"reports_to":"reports_count","age":"average_age"},inplace=True)
    return f_df """

#Employees with missing information
#https://leetcode.com/problems/employees-with-missing-information/description/

""" import pandas as pd

def find_employees(employees: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    new_df=employees.merge(salaries,how="outer",on="employee_id")
    return new_df[new_df.isna().any(axis=1)][["employee_id"]].sort_values(by="employee_id") """

#Employees whose manager left the company
#https://leetcode.com/problems/employees-whose-manager-left-the-company/description/

""" df = employees[employees['salary']<30000].dropna(subset='manager_id')
    return df[~df['manager_id'].isin(employees['employee_id'])][['employee_id']].sort_values('employee_id')
 """

#Medium problems

#Second highest salary
#https://leetcode.com/problems/second-highest-salary/
""" 
import pandas as pd

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    unique_salaries = employee['salary'].drop_duplicates()

    second_highest = unique_salaries.nlargest(2).iloc[-1] if len(unique_salaries) >= 2 else None

    if second_highest is None:
        return pd.DataFrame({'SecondHighestSalary': [None]})

    result_df = pd.DataFrame({'SecondHighestSalary': [second_highest]})

    return result_df """

#Department highest salary
#https://leetcode.com/problems/department-highest-salary/
""" 
import pandas as pd

def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    if employee.empty or department.empty:
        return pd.DataFrame(columns=['Department','Employee', 'Salary'])
    
    merged_df = employee.merge(department, left_on='departmentId', right_on='id', suffixes=('_employee', '_department'))
    
    highest_salary_df = merged_df.groupby('departmentId').apply(lambda x: x[x['salary'] == x['salary'].max()])
    
    highest_salary_df = highest_salary_df.reset_index(drop=True)
    
    result_df = highest_salary_df[['name_department', 'name_employee', 'salary']]
    
    result_df.columns = ['Department','Employee', 'Salary']
    
    return result_df """

#https://leetcode.com/problems/investments-in-2016/description/
#Investments in 2016

""" import pandas as pd

def find_investments(insurance: pd.DataFrame) -> pd.DataFrame:
    uniq_lat_lon = insurance.drop_duplicates(subset = ['lat','lon'], keep = False).pid
    not_uniq_tiv_2015 = insurance.loc[insurance.duplicated(subset = 'tiv_2015', keep=False)].pid
    df = insurance.loc[insurance.pid.isin(uniq_lat_lon) & insurance.pid.isin(not_uniq_tiv_2015)]
    return df[['tiv_2016']].sum().to_frame('tiv_2016').round(2) """

#https://leetcode.com/problems/rank-scores/description/
#Ranking scores

""" import pandas as pd

def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    scores=scores.sort_values(by=["score"],ascending=False)
    scores["rank"]=scores["score"].rank(method="dense",ascending=False)
    scores["rank"]=scores["rank"].astype("int64")
    return scores[["score","rank"]] """

#https://leetcode.com/problems/capital-gainloss/
#Capital gain/loss

""" import pandas as pd

def capital_gainloss(stocks: pd.DataFrame) -> pd.DataFrame:
    buy_df=stocks[stocks["operation"]=="Buy"]
    buy_df.rename(columns={"price":"buy_price"},inplace=True)
    sell_df=stocks[stocks["operation"]=="Sell"]
    sell_df.rename(columns={"price":"sell_price"},inplace=True)
    f_df=sell_df.merge(buy_df,how="inner",on="stock_name")
    f_df["capital_gain_loss"]=f_df["price_x"]-f_df["price_y"]
    return f_df.groupby("stock_name")["capital_gain_loss"].sum().reset_index() """

#Tree node
#https://leetcode.com/problems/tree-node/

""" import pandas as pd

def tree_node(tree: pd.DataFrame) -> pd.DataFrame:
    l=list(tree["id"].isin(tree["p_id"]))
    l2=list(tree["p_id"].isna())
    l3=list(zip(l,l2))
    l4=[]
    for t in l3:
        if t[1]==True:
            l4.append("Root")
        elif t[0]==True:
            l4.append("Inner")
        else:
            l4.append("Leaf")
    tree["type"]=pd.Series(l4)
    return tree[["id","type"]] """

#https://leetcode.com/problems/movie-rating/
#Movie rating

""" import pandas as pd

def movie_rating(movies: pd.DataFrame, users: pd.DataFrame, movie_rating: pd.DataFrame) -> pd.DataFrame:
    #Getting the first result
    count_ratings=movie_rating.groupby("user_id")["rating"].count().reset_index()
    f1_df=count_ratings.merge(users,how="inner",on="user_id")
    f1_df.sort_values(by=["rating","name"],ascending=[False,True],ignore_index=True)
    #Getting the second result
    movie_rating["month"]=pd.DatetimeIndex(movie_rating["created_at"]).month
    new_df=movie_rating[movie_rating["month"]==2]
    f2_df=new_df.groupby("movie_id")["rating"].mean().reset_index()
    f2_df=f2_df.merge(movies,how="inner",on="movie_id")
    f2_df.sort_values(by=["rating","title"],ascending=[False,True],inplace=True,ignore_index=True)
    final=pd.DataFrame({"results":[f1_df.loc[0,"name"],f2_df.loc[0,"title"]]})
    return final """

#https://leetcode.com/problems/restaurant-growth/
#Restaurant growth

""" import pandas as pd

def restaurant_growth(customer: pd.DataFrame) -> pd.DataFrame:
    df = customer.sort_values("visited_on").groupby("visited_on")[["amount"]].sum()
    df = df.assign(amount = df.rolling("7D").sum(), average_amount = round(df.rolling("7D").sum()/7,2))
    return df.loc[df.index >= df.index.min() + pd.DateOffset(6)].reset_index() """

#https://leetcode.com/problems/consecutive-numbers/
#Consecutive numbers

#Partial solution (when the dataframe contains only one number with three consecutive app)
""" import pandas as pd

def consecutive_numbers(logs: pd.DataFrame) -> pd.DataFrame:
    def consecutiveAppearance(s):
        counter=1
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                pointer1=s[i]
                pointer2=s[j]
                while pointer1==pointer2:
                    counter=counter+1
                    if counter==3:
                        return pointer1
                    else:pass
    result=consecutiveAppearance(logs["num"])
    f_df=pd.DataFrame({"ConsecutiveNums":[result]})
    return f_df """
#Solution
""" import pandas as pd

def consecutive_numbers(logs: pd.DataFrame) -> pd.DataFrame:
    logs['var'] = logs.num.rolling(window=3).var()

    return pd.DataFrame(data = {'ConsecutiveNums' : logs.query('var == 0').num.unique()}) """

#https://leetcode.com/problems/biggest-single-number/?envType=study-plan-v2&envId=top-sql-50
#Biggest single number

""" import pandas as pd

def biggest_single_number(my_numbers: pd.DataFrame) -> pd.DataFrame:
    df2=my_numbers.groupby("num").value_counts().reset_index()
#    df2.rename(columns={0:"cnt"},inplace=True)
    element=df2[df2["count"]==1].sort_values(by=["num"],ascending=False).iloc[0,0]
    f_df=pd.DataFrame({"num":[element]})
    return f_df """

#https://leetcode.com/problems/count-salary-categories/?envType=study-plan-v2&envId=top-sql-50
#Salary categories

""" import pandas as pd

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    low=accounts[accounts["income"]<20000]["income"].count()
    average=accounts[(accounts["income"]>=20000)&(accounts["income"]<=50000)]["income"].count()
    high=accounts[accounts["income"]>50000]["income"].count()
    f_df=pd.DataFrame({"category":["Low Salary","Average Salary","High Salary"],"accounts_count":[low,average,high]})
    return f_df """

#https://leetcode.com/problems/department-top-three-salaries/?envType=study-plan-v2&envId=top-sql-50
#Department top three salaries

""" import pandas as pd

def top_three_salaries(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    employee.sort_values(by=["departmentId","salary"],inplace=True,ignore_index=True,ascending=[True,False])
    employee["sal_rnk"]=employee.groupby("departmentId")["salary"].rank(method="dense",ascending=False)
    f_df=employee[employee["sal_rnk"].isin([1,2,3])]
    f_df=f_df.merge(department,how="inner",left_on="departmentId",right_on="id")
    return f_df[["department","name","salary"]] """

#https://leetcode.com/problems/human-traffic-of-stadium/
#Humman traffic on stadium

#Partial solution

""" import pandas as pd

def human_traffic(stadium: pd.DataFrame) -> pd.DataFrame:
    def consecutive_numbers(s):
        idx=[]
        i=0
        for n in s:
            if n>100:
                idx.append(s.index(n))
                i=i+1
            else:
                idx=[]
                i=0
        if i>=3:
            return idx
        else:
            idx=[]
            return idx
    l=consecutive_numbers(stadium["people"].to_list())
    l2=[val+1 for val in l]
    return stadium[stadium["id"].isin(l2)] """

#Solution

""" import pandas as pd

def human_traffic(stadium: pd.DataFrame) -> pd.DataFrame:
    D=stadium.sort_values("id").query("people>=100").reset_index(drop=1)
    return D.assign(c=D.groupby(D.id-D.index).id.transform("size")).query("c>2").iloc[:,:3] """

#https://leetcode.com/problems/trips-and-users/description/
#Trips and users

""" import pandas as pd

def trips_and_users(trips: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    users = users[users.banned == 'No']
    trips = trips[trips.request_at
                     .between('2013-10-01','2013-10-03')
                    ].rename(columns = {'request_at': 'Day'})

    trips['can'] = trips.status.str.startswith('can')

    trips = trips[(trips.client_id.isin(users.users_id)) &
                  (trips.driver_id.isin(users.users_id))
                  ].groupby('Day')['can'].agg(['sum','size']).reset_index()

    trips['Cancellation Rate'] = (trips['sum']/trips['size']).round(2)
    
    return trips[['Day','Cancellation Rate']] """

#https://leetcode.com/problems/managers-with-at-least-5-direct-reports/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
#Manager with at least 5 direct reports

""" import pandas as pd

def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    new_df=employee.groupby("managerId")["department"].count().reset_index()
    f_df=new_df[new_df["department"]>=5]
    return employee[employee["id"].isin(f_df["managerId"])][["name"]] """

#https://leetcode.com/problems/nth-highest-salary/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata
#Nth highest salary

""" import pandas as pd

def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    employee.sort_values(by=["salary"],inplace=True,ascending=False,ignore_index=True)
    employee.drop_duplicates(subset="salary",inplace=True,ignore_index=True,keep="first")
    if N>len(employee):
        f_df=pd.DataFrame({f"getNthHighestSalary({N})":[None]})
        return f_df
    else:
        nth_val=employee.loc[N-1,"salary"]
        f_df=pd.DataFrame({f"getNthHighestSalary({N})":[nth_val]})
        return f_df """

#https://techtfq.com/blog/practice-writing-sql-queries-using-real-dataset

""" import pandas as pd

#Problem_1
#How many olympic games have been held?

len(df.drop_duplicates(subset=["Games"]))

#Problem_2
#List down all the olympic games that have been held so far

df.drop_duplicates(subset=["Games"],keep="first")["Games"]

#Problem_3
#Mention the total number of nations that participate in each olympic game

s=df.drop_duplicates(subset=["Games","Year"],keep="first")["Year"]
s.dropna()
new_df=df.groupby("Year")
l=[]
for year in s:
  l.append(new_df.get_group(year).drop_duplicates(subset=["Team"],keep="first")["Team"].count())
final_df=pd.DataFrame({"year":s,"number_of_nations":l})
final_df.sort_values(by=["year"],ignore_index=True,inplace=True)

#Problem_4
#Which year saw the greatest and lowest number of countries participating in olympics?

f_df[f_df["Team"]==f_df["Team"].max()]["Games"]
f_df[f_df["Team"]==f_df["Team"].min()]["Games"]

#Problem_5
#Which nation has participated in all the olympic games?

df.groupby("Team")["Games"].count().reset_index().sort_values(by=["Games"],ascending=False)
#No one has been in all olympic games

#Problem_6
#Identify the sport that was played in all summer olympics

len(df[df["Season"]=="Summer"])
n_df=df[df["Season"]=="Summer"].groupby("Sport")["ID"].count().reset_index()
n_df.sort_values(by=["ID"],ascending=False,inplace=True)
n_df[n_df["ID"]==222552]
#No sport has been played in all summer olympics """

#Brands (SQL interview question)
""" 
df["new_col"]=df.groupby("brand")["amount"].shift(fill_value=0)
new_df=df[df["year"]>2018]
new_df["subt"]=df["amount"]-df["new_col"]
f_df=new_df[new_df["subt"]>0]
final=f_df.groupby("brand")["subt"].count().reset_index()
final[final["subt"]==(max(df["year"]-min(df["year"])))] """

#Meaningful message (SQL interview question)

""" import pandas as pd

l=[]
for i in range(len(df)):
  if df.loc[i,"translation"]!=None:
    l.append(df.loc[i,"translation"])
  else:
    l.append(df.loc[i,"comment"])
df["meaningful_message"]=l
df["meaningful_message"] """

#Teams
""" 
import pandas as pd

df2=df.copy()
f_df=df.merge(df2,how="cross")
f_df.rename(columns={"team_name_x":"team","team_name_y":"opponent"},inplace=True)
new_df=f_df[f_df["team"]!=f_df["opponent"]][["team","opponent"]]
new_df2=new_df.copy()
final_df=pd.concat([new_df,new_df2],ignore_index=True,axis=0)
final_df.sort_values(by=["team","opponent"],inplace=True)
 """

#Records
""" import pandas as pd

df1=source[~(source["id"].isin(target["id"]))]
df2=source[(source["id"].isin(target["id"]))&(~(source["name"].isin(target["name"])))]
df3=target[~(target["id"].isin(source["id"]))]
final_df=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
final_df.loc[0,"name"]="new in source"
final_df.loc[1,"name"]="mismatch"
final_df.loc[2,"name"]="new in target" """