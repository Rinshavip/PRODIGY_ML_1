import pandas as sph
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

#Reading Training csv dataset using pandas
try:
    #Taking necessary columns from dataset
    tds = sph.read_csv("train.csv", usecols=["LotArea", "SalePrice","FullBath","BedroomAbvGr"])
except FileNotFoundError:
     print("File not found.")
     exit()

#Training The model
x = tds[["LotArea","FullBath","BedroomAbvGr"]]
y = tds["SalePrice"]
linreg.fit(x,y)

#Creating a new panda frame and testing a new test data and saving it, for larger predictions
tst = sph.read_csv("test.csv", usecols=["LotArea","FullBath","BedroomAbvGr"])
tpv = linreg.predict(tst)
tst["SalePrice"] = tpv
tst.to_csv('test_result.csv',index=False)