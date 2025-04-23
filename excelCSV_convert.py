import pandas as pd
read_file = pd.read_excel('cashflows_2024.xlsx')

read_file.to_csv ('selectionscashflows_2024.csv',  
                  index = None, 
                  header=True) 
    
# read csv file and convert  
# into a dataframe object 
df = pd.DataFrame(pd.read_csv('cashflows_2024.csv')) 
  
# show the dataframe 
df