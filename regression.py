X = 128,256,256,128,512,256,256,128,128,128,256,256,128,128,128,256
Y = 14999000,19799100,24999000,14999000,26999000,15999000,15999000,2499000,2499000,2799000,5999000,5999000,2799000,2499000,2999000,5399000
#library
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
#dataset 
X= np.array([128,256,256,128,512,256,256,128,128,128,256,256,128,128,128,256]).reshape((-1, 1))
Y= np.array([14999000,19799100,24999000,14999000,26999000,15999000,15999000,2499000,2499000,2799000,5999000,5999000,2799000,2499000,2999000,5399000])
#call model regression
model = LinearRegression().fit(X,Y)
#save model
filename = 'model.sav'
joblib.dump(model, filename)
#load model
loaded_model = joblib.load(filename)
#prediction model
loaded_model.predict(128)