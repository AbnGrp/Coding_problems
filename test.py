import pandas as pd

def find_employees(employee: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,len(employee)+1):
        if employee.loc[i,"salary"]>employee.loc[employee["managerId"],"salary"]:
            return employee[i,"name"]