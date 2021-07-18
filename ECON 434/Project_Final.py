#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
#import os


# In[1]:


#Read data
#os.chdir("C:\\Users\\mmspe\\OneDrive\\Documents\\Python Scripts\\434")
data = pd.read_csv("Uber_dataset.csv")

#Add logs
data["log_popestimate"] = np.log(data["popestimate"])
data["log_employment"] = np.log(data["employment"])
data["log_aveFareTotal"] = np.log(data["aveFareTotal"])
data["log_VRHTotal"] = np.log(data["VRHTotal"])
data["log_VOMSTotal"] = np.log(data["VOMSTotal"])
data["log_VRMTotal"] = np.log(data["VRMTotal"])
data["log_gasPrice"] = np.log(data["gasPrice"])

#Drop nas
data = data.dropna()

#Dependant variable
Y = np.log(data["UPTTotal"])
#First of two variations of variable of interest
uber_dummy = np.array(data["treatUberX"], ndmin = 2).T
#Second of two variations of variable of interest
uber_pen = np.array(data["treatGTNotStd"], ndmin = 2).T

#Vectorize controls for matrix multiplication
lnpop = np.array(data["log_popestimate"], ndmin = 2).T
lnemp = np.array(data["log_employment"], ndmin = 2).T
lnfare = np.array(data["log_aveFareTotal"], ndmin = 2).T
lnvhours = np.array(data["log_VRHTotal"], ndmin = 2).T
lnnumv = np.array(data["log_VOMSTotal"], ndmin = 2).T
lnmiles = np.array(data["log_VRMTotal"], ndmin = 2).T
lngas = np.array(data["log_gasPrice"], ndmin = 2).T

controls =  np.concatenate((lnpop, lnemp, lnfare, lnvhours, lnnumv, 
                                  lnmiles, lngas), axis = 1)
#Number of samples
n = len(data) #58354 samples
#Constant
cons = np.ones([n ,1])


# In[15]:


#### Regressions ####
#   Regression 1

#       a) D = dummy

X1a = np.concatenate((cons, uber_dummy, controls), axis = 1)

betahat_1a = np.linalg.inv(X1a.T @ X1a) @ (X1a.T @ Y)
ehat_1a = Y - X1a @ betahat_1a
ehat_1a = np.array(ehat_1a, ndmin = 2).T

Sigmahat_1a = (X1a * ehat_1a).T @ (X1a * ehat_1a) / n

Qhat_1a = np.linalg.inv(X1a.T @ X1a / n)
Vhat_1a = Qhat_1a @ Sigmahat_1a @ Qhat_1a
sdhat_1a = np.sqrt(Vhat_1a[1 ,1]) / np.sqrt(n)
cil_1a = betahat_1a[1] - 1.96 * sdhat_1a; cir_1a = betahat_1a[1] + 1.96 * sdhat_1a

#       b) D = search intensity

X1b = np.concatenate((cons, uber_pen, controls), axis = 1)

betahat_1b = np.linalg.inv(X1b.T @ X1b) @ (X1b.T @ Y)
ehat_1b = Y - X1b @ betahat_1b
ehat_1b = np.array(ehat_1b, ndmin = 2).T

Sigmahat_1b = (X1b * ehat_1b).T @ (X1b * ehat_1b) / n

Qhat_1b = np.linalg.inv(X1b.T @ X1b / n)
Vhat_1b = Qhat_1b @ Sigmahat_1b @ Qhat_1b
sdhat_1b = np.sqrt(Vhat_1b[1 ,1]) / np.sqrt(n)
cil_1b = betahat_1b[1] - 1.96 * sdhat_1b; cir_1b = betahat_1b[1] + 1.96 * sdhat_1b



print('In OLS model 1:\n')
print("The coefficient for 'treatUberX' is ", betahat_1a[1])
print("Standard Error is ", sdhat_1a)
print("95% confidence interval is ["+str(cil_1a)+","+str(cir_1a)+"]")

print("\nThe coefficient for 'treatGTNotStd' is ", betahat_1b[1])
print("Standard Error is ", sdhat_1b)
print("95% confidence interval is ["+str(cil_1b)+","+str(cir_1b)+"]")


# In[22]:


#   Regression 2
#𝛾i is a transit agency specific fixed effect; 𝛿t is a year–month specific fixed effect

#       a) D = dummy

#Create dummies for transit agency fixed effects
agency_dummies = pd.get_dummies(data["agency"])
yrmon_dummies  = pd.get_dummies(data["dateSurvey"])


# In[23]:


X2a = np.concatenate((uber_dummy, agency_dummies, yrmon_dummies, controls), axis = 1)

betahat_2a = np.linalg.inv(X2a.T @ X2a) @ (X2a.T @ Y)
ehat_2a = Y - X2a @ betahat_2a
ehat_2a = np.array(ehat_2a, ndmin = 2).T

Sigmahat_2a = (X2a * ehat_2a).T @ (X2a * ehat_2a) / n

Qhat_2a = np.linalg.inv(X2a.T @ X2a / n)
Vhat_2a = Qhat_2a @ Sigmahat_2a @ Qhat_2a
sdhat_2a = np.sqrt(Vhat_2a[0 ,0]) / np.sqrt(n)
cil_2a = betahat_2a[0] - 1.96 * sdhat_2a; cir_2a = betahat_2a[0] + 1.96 * sdhat_2a

#       b) D = search intensity

X2b = np.concatenate((uber_pen, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_2b = np.linalg.inv(X2b.T @ X2b) @ (X2b.T @ Y)
ehat_2b = Y - X2b @ betahat_2b
ehat_2b = np.array(ehat_2b, ndmin = 2).T

Sigmahat_2b = (X2b * ehat_2b).T @ (X2b * ehat_2b) / n

Qhat_2b = np.linalg.inv(X2b.T @ X2b / n)
Vhat_2b = Qhat_2b @ Sigmahat_2b @ Qhat_2b
sdhat_2b = np.sqrt(Vhat_2b[0 ,0]) / np.sqrt(n)
cil_2b = betahat_2b[0] - 1.96 * sdhat_2b; cir_2b = betahat_2b[0] + 1.96 * sdhat_2b



print('In OLS model 2(including time and location effect):\n')
print("The coefficient for 'treatUberX' is ", betahat_2a[0])
print("Standard Error is ", sdhat_2a)
print("95% confidence interval is ["+str(cil_2a)+","+str(cir_2a)+"]")

print("\nThe coefficient for 'treatGTNotStd' is ", betahat_2b[0])
print("Standard Error is ", sdhat_2b)
print("95% confidence interval is ["+str(cil_2b)+","+str(cir_2b)+"]")


# In[3]:


#   Regression 3-----IMPORTANT

#   a) D = dummy

#Calculate median population: 1304926
median_pop = np.median(data["popestimate"])
#Create dummy
data["pop_med_dummy"] = (data["popestimate"] > median_pop).astype(int)
#Create interaction


# In[35]:


data["pop_med_int"] = data["pop_med_dummy"] * data["treatUberX"]
#pop_med_dum = np.array(data["pop_med_dummy"], ndmin = 2).T
pop_med_int = np.array(data["pop_med_int"], ndmin = 2).T


# In[ ]:


X3a = np.concatenate((uber_dummy, pop_med_int, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_3a = np.linalg.inv(X3a.T @ X3a) @ (X3a.T @ Y)
ehat_3a = Y - X3a @ betahat_3a
ehat_3a = np.array(ehat_3a, ndmin = 2).T

Sigmahat_3a = (X3a * ehat_3a).T @ (X3a * ehat_3a) / n

Qhat_3a = np.linalg.inv(X3a.T @ X3a / n)
Vhat_3a = Qhat_3a @ Sigmahat_3a @ Qhat_3a
sdhat_3a = np.sqrt(Vhat_3a[0 ,0]) / np.sqrt(n)
cil_3a = betahat_3a[0] - 1.96 * sdhat_3a; cir_3a = betahat_3a[0] + 1.96 * sdhat_3a

sdhat_3a_pop = np.sqrt(Vhat_3a[1 ,1]) / np.sqrt(n)

cil_3a_pop = betahat_3a[1] - 1.96 * sdhat_3a_pop
cir_3a_pop = betahat_3a[1] + 1.96 * sdhat_3a_pop


# In[36]:


#   b) D = search intensity

#Create interaction
data["pop_med_int_pen"] = data["pop_med_dummy"] * data["treatGTNotStd"]
pop_med_int_pen = np.array(data["pop_med_int_pen"], ndmin = 2).T


# In[25]:


X3b = np.concatenate((uber_pen, pop_med_int_pen, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_3b = np.linalg.inv(X3b.T @ X3b) @ (X3b.T @ Y)
ehat_3b = Y - X3b @ betahat_3b
ehat_3b = np.array(ehat_3b, ndmin = 2).T

Sigmahat_3b = (X3b * ehat_3b).T @ (X3b * ehat_3b) / n

Qhat_3b = np.linalg.inv(X3b.T @ X3b / n)
Vhat_3b = Qhat_3b @ Sigmahat_3b @ Qhat_3b
sdhat_3b = np.sqrt(Vhat_3b[0 ,0]) / np.sqrt(n)
cil_3b = betahat_3b[0] - 1.96 * sdhat_3b; cir_3b = betahat_3b[0] + 1.96 * sdhat_3b

sdhat_3b_pop = np.sqrt(Vhat_3b[1 ,1]) / np.sqrt(n)

cil_3b_pop = betahat_3b[1] - 1.96 * sdhat_3b_pop
cir_3b_pop = betahat_3b[1] + 1.96 * sdhat_3b_pop




pop_OLS = pd.DataFrame([[betahat_3a[0],betahat_3a[1],betahat_3b[0],betahat_3b[1]],
                          [sdhat_3a,sdhat_3a_pop, sdhat_3b,sdhat_3b_pop]], 
                          columns=['Uber_dummy','Above_median_pop*Uber_dummy',
                                   'Uber_pen', 'Above_median_pop*Uber_pen'],
                         index=['coef','SE'])

pop_OLS


# In[37]:


#   Regression 4----IMPORTANT

#Calculate median rides: 
median_rides = np.median(data["UPTTotal"])
#Create dummy
data["rides_med_dummy"] = (data["UPTTotal"] > median_rides).astype(int)
#Create interaction
data["rides_med_int"] = data["rides_med_dummy"] * data["treatUberX"]
#rides_med_dum = np.array(data["rides_med_dummy"], ndmin = 2).T
rides_med_int = np.array(data["rides_med_dummy"], ndmin = 2).T


# In[ ]:


X4a = np.concatenate((uber_dummy, rides_med_int, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_4a = np.linalg.inv(X4a.T @ X4a) @ (X4a.T @ Y)
ehat_4a = Y - X4a @ betahat_4a
ehat_4a = np.array(ehat_4a, ndmin = 2).T

Sigmahat_4a = (X4a * ehat_4a).T @ (X4a * ehat_4a) / n

Qhat_4a = np.linalg.inv(X4a.T @ X4a / n)
Vhat_4a = Qhat_4a @ Sigmahat_4a @ Qhat_4a
sdhat_4a = np.sqrt(Vhat_4a[0 ,0]) / np.sqrt(n)
cil_4a = betahat_4a[0] - 1.96 * sdhat_4a; cir_4a = betahat_4a[0] + 1.96 * sdhat_4a

sdhat_4a_rides = np.sqrt(Vhat_4a[1 ,1]) / np.sqrt(n)

cil_4a_rides = betahat_4a[1] - 1.96 * sdhat_4a_rides
cir_4a_rides = betahat_4a[1] + 1.96 * sdhat_4a_rides


# In[38]:


#   b) D = search intensity

data["rides_med_int_pen"] = data["rides_med_dummy"] * data["treatGTNotStd"]
rides_med_int_pen = np.array(data["rides_med_int_pen"], ndmin = 2).T


# In[27]:


X4b = np.concatenate((uber_pen, rides_med_int_pen, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_4b = np.linalg.inv(X4b.T @ X4b) @ (X4b.T @ Y)
ehat_4b = Y - X3b @ betahat_4b
ehat_4b = np.array(ehat_4b, ndmin = 2).T

Sigmahat_4b = (X4b * ehat_4b).T @ (X4b * ehat_4b) / n

Qhat_4b = np.linalg.inv(X4b.T @ X4b / n)
Vhat_4b = Qhat_4b @ Sigmahat_4b @ Qhat_4b
sdhat_4b = np.sqrt(Vhat_4b[0 ,0]) / np.sqrt(n)
cil_4b = betahat_4b[0] - 1.96 * sdhat_4b; cir_4b = betahat_4b[0] + 1.96 * sdhat_4b

sdhat_4b_rides = np.sqrt(Vhat_4b[1 ,1]) / np.sqrt(n)

cil_4b_rides = betahat_4b[1] - 1.96 * sdhat_4b_rides
cir_4b_rides = betahat_4b[1] + 1.96 * sdhat_4b_rides





rides_OLS = pd.DataFrame([[betahat_4a[0],betahat_4a[1],betahat_4b[0],betahat_4b[1]],
                          [sdhat_4a,sdhat_4a_rides, sdhat_4b,sdhat_4b_rides]], 
                          columns=['Uber_dummy','Above_median_rides*Uber_dummy',
                                   'Uber_pen', 'Above_median_rides*Uber_pen'],
                         index=['coef','SE'])

rides_OLS


# In[49]:


#   Regression 5
from sklearn.linear_model import LassoCV

#   a) D = dummy

#Rescale controls
muhat_scale = np.mean(controls,axis = 0)
stdhat_scale = np.std(controls,axis = 0)
controls_scaled = (controls - muhat_scale )/ stdhat_scale


# In[ ]:


X5a = np.concatenate((uber_dummy, pop_med_int, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)


#run lasso
lasso5a = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lasso5a.fit(X5a ,Y)
coef5a = lasso5a.coef_
sel5a = (coef5a != 0)


# In[50]:


#   b) D = search intensity

#Rescale
muhat_scale_pen = np.mean(uber_pen)
stdhat_scale_pen = np.std(uber_pen)
uber_pen_scaled = (uber_pen - muhat_scale_pen) / stdhat_scale_pen


# In[7]:


X5b = np.concatenate((uber_pen_scaled, pop_med_int_pen, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)

lasso5b = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lasso5b.fit(X5b, Y)
coef5b = lasso5b.coef_
sel5b = (coef5b != 0)




pop_lasso = pd.DataFrame([[coef5a[0],coef5a[1],coef5b[0],coef5b[1]]],
                         columns=['lasso_Uber_dummy','lasso_Above_med_pop*Uber_dummy',
                                   'lasso_Uber_pen', 'lasso_Above_med_pop*Uber_pen'],
                          index=['coef'])

pop_lasso


# In[8]:


#   Regression 6

#   a) D = dummy

X6a = np.concatenate((uber_dummy, rides_med_int, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)

#run lasso
lasso6a = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lasso6a.fit(X6a, Y)
coef6a = lasso6a.coef_
sel6a = (coef6a != 0)


#   b) D = search intensity

X6b = np.concatenate((uber_pen_scaled, rides_med_int_pen, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)

lasso6b = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lasso6b.fit(X6b, Y)
coef6b = lasso6b.coef_




rides_lasso = pd.DataFrame([[coef6a[0],coef6a[1],coef6b[0],coef6b[1]]],
                           columns=['lasso_Uber_dummy','lasso_Above_med_pop*Uber_dummy',
                                   'lasso_Uber_pen', 'lasso_Above_med_pop*Uber_pen'],
                          index=['coef'])

rides_lasso


# In[9]:


#   Regression 7---DOUBLE LASSO

#  Regresion 7.1 --- For population(corresponding to regression 5)
#from sklearn.linear_model import MultiTaskLassoCV

#       a) D= dummy
#1st stage--same as regression 5 
coef7ia_1 = lasso5a.coef_.copy()

#2nd stage
X7ia = np.concatenate((agency_dummies, yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_dummy 
lasso7ia_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7ia, uber_dummy)
coef7ia_21 = lasso7ia_21.coef_
ehat7ia_21 = uber_dummy.T- coef7ia_21.T @ X7ia.T

#fit on pop_med_int
lasso7ia_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7ia, pop_med_int)
coef7ia_22 = lasso7ia_22.coef_
ehat7ia_22 = pop_med_int.T- coef7ia_22.T @ X7ia.T

#Calculate alpha
alpha7ia_B1 = (np.array(Y - X7ia @ coef7ia_1[2:]) 
              @ ehat7ia_21.T) @ np.linalg.inv(uber_dummy.T @ (ehat7ia_21).T) 

alpha7ia_B2 = (np.array(Y - X7ia @ coef7ia_1[2:]) 
              @ ehat7ia_22.T) @ np.linalg.inv(pop_med_int.T @ (ehat7ia_22).T) 


#       b) D = search intensity

#1st stage--same as regression 5 
coef7ib_1 = lasso5b.coef_.copy()

#2nd stage
X7ib = np.concatenate((agency_dummies,yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_pen
lasso7ib_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7ib, uber_pen)
coef7ib_21 = lasso7ib_21.coef_
ehat7ib_21 = uber_pen.T- coef7ib_21.T @ X7ib.T

#fit on pop_med_int_pen
lasso7ib_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7ib, pop_med_int_pen)
coef7ib_22 = lasso7ib_22.coef_
ehat7ib_22 = pop_med_int_pen.T- coef7ib_22.T @ X7ib.T

#Calculate alpha
alpha7ib_B1 = (np.array(Y - X7ib @ coef7ib_1[2:]) 
              @ ehat7ib_21.T) @ np.linalg.inv(uber_pen.T @ (ehat7ib_21).T) 

alpha7ib_B2 = (np.array(Y - X7ib @ coef7ib_1[2:]) 
              @ ehat7ib_22.T) @ np.linalg.inv(pop_med_int_pen.T @ (ehat7ib_22).T) 




pop_dblasso = pd.DataFrame([[alpha7ia_B1,alpha7ia_B2,alpha7ib_B1,alpha7ib_B2]],
                         columns=['dblasso_Uber_dummy','dblasso_Above_med*Uber_dummy',
                                   'dblasso_Uber_pen', 'dblasso_Above_med*Uber_pen'],
                          index=['coef'])

pop_dblasso


# In[10]:


#  Regresion 7.2 --- For rides(corresponding to regression 6)

#       a) D= dummy
#1st stage--same as regression 6
coef7iia_1 = lasso6a.coef_.copy()

#2nd stage
X7iia = np.concatenate((agency_dummies,yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_dummy 
lasso7iia_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7iia,uber_dummy)
coef7iia_21 = lasso7iia_21.coef_
ehat7iia_21 = uber_dummy.T- coef7iia_21.T @ X7iia.T

#fit on rides_med_int
lasso7iia_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7iia,rides_med_int)
coef7iia_22 = lasso7iia_22.coef_
ehat7iia_22 = rides_med_int.T - coef7iia_22.T @ X7iia.T

#Calculate alpha
alpha7iia_B1 = (np.array(Y - X7iia @ coef7iia_1[2:]) 
              @ ehat7iia_21.T) @ np.linalg.inv(uber_dummy.T @ (ehat7iia_21).T) 

alpha7iia_B2 = (np.array(Y - X7iia @ coef7iia_1[2:]) 
              @ ehat7iia_22.T) @ np.linalg.inv(rides_med_int.T @ (ehat7iia_22).T) 


#       b) D = search intensity

#1st stage--same as regression 6 
coef7iib_1 = lasso6b.coef_.copy()

#2nd stage
X7iib = np.concatenate((agency_dummies,yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_pen
lasso7iib_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7iib, uber_pen)
coef7iib_21 = lasso7iib_21.coef_
ehat7iib_21 = uber_pen.T - coef7iib_21.T @ X7iib.T

#fit on rides_med_int_pen
lasso7iib_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=10000).fit(X7iib, rides_med_int_pen)
coef7iib_22 = lasso7iib_22.coef_
ehat7iib_22 = rides_med_int_pen.T - coef7iib_22.T @ X7iib.T

#Calculate alpha
alpha7iib_B1 = (np.array(Y - X7iib @ coef7iib_1[2:]) 
              @ ehat7iib_21.T) @ np.linalg.inv(uber_pen.T @ (ehat7iib_21).T) 

alpha7iib_B2 = (np.array(Y - X7iib @ coef7iib_1[2:]) 
              @ ehat7iib_22.T) @ np.linalg.inv(rides_med_int_pen.T @ (ehat7iib_22).T) 




rides_dblasso = pd.DataFrame([[alpha7iia_B1,alpha7iia_B2,alpha7iib_B1,alpha7iib_B2]],
                         columns=['dblasso_Uber_dummy','dblasso_Above_med*Uber_dummy',
                                   'dblasso_Uber_pen', 'dblasso_Above_med*Uber_pen'],
                          index=['coef'])

rides_dblasso


# In[58]:


#   Regression 8
from sklearn.preprocessing import PolynomialFeatures

#Create interactions
pol_int = PolynomialFeatures(degree=5, include_bias=False)
int_controls = pol_int.fit_transform(controls)

muhat_scale_int = np.mean(int_controls,axis = 0)
stdhat_scale_int = np.std(int_controls,axis = 0)
int_controls_scaled = (int_controls - muhat_scale_int )/ stdhat_scale_int


# In[11]:


#   a) D = dummy

X8a = np.concatenate((uber_dummy, pop_med_int, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)

lasso8a = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lasso8a.fit(X8a ,Y)
coef8a = lasso8a.coef_
sel8a = (coef8a != 0)

#   b) D = search intensity

X8b = np.concatenate((uber_pen, pop_med_int_pen, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)

lasso8b = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lasso8b.fit(X8b, Y)
coef8b = lasso8b.coef_
sel8b = (coef8b != 0)



pop_poly_lasso = pd.DataFrame([[coef8a[0],coef8a[1],coef8b[0],coef8b[1]]],
                         columns=['lasso_Uber_dummy','lasso_Above_med_pop*Uber_dummy',
                                   'lasso_Uber_pen', 'lasso_Above_med_pop*Uber_pen'],
                          index=['coef'])

pop_poly_lasso


# In[12]:


#   Regression 9

#   a) D = dummy

X9a = np.concatenate((uber_dummy, rides_med_int, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)


lasso9a = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lasso9a.fit(X9a ,Y)
coef9a = lasso9a.coef_
sel9a = (coef9a != 0)

#   b) D = search intensity

X9b = np.concatenate((uber_pen, rides_med_int_pen, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)

lasso9b = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lasso9b.fit(X9b ,Y)
coef9b = lasso9b.coef_
sel9b = (coef9b != 0)



rides_poly_lasso = pd.DataFrame([[coef9a[0],coef9a[1],coef9b[0],coef9b[1]]],
                           columns=['lasso_Uber_dummy','lasso_Above_med_pop*Uber_dummy',
                                   'lasso_Uber_pen', 'lasso_Above_med_pop*Uber_pen'],
                          index=['coef'])

rides_poly_lasso


# In[13]:


#   Regression 10 ---DOUBLE LASSO

#  Regresion 10.1 --- For population(corresponding to regression 8)

#       a) D= dummy
#1st stage--same as regression 8 
coef10ia_1 = lasso8a.coef_.copy()

#2nd stage
X10ia = np.concatenate((agency_dummies, yrmon_dummies, int_controls_scaled), axis = 1)

#fit on uber_dummy 
lasso10ia_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=100000).fit(X10ia, uber_dummy)
coef10ia_21 = lasso10ia_21.coef_
ehat10ia_21 = uber_dummy.T- coef10ia_21.T @ X10ia.T

#fit on pop_med_int
lasso10ia_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=100000).fit(X10ia, pop_med_int)
coef10ia_22 = lasso10ia_22.coef_
ehat10ia_22 = pop_med_int.T- coef10ia_22.T @ X10ia.T

#Calculate alpha
alpha10ia_B1 = (np.array(Y - X10ia @ coef10ia_1[2:]) 
              @ ehat10ia_21.T) @ np.linalg.inv(uber_dummy.T @ (ehat10ia_21).T) 

alpha10ia_B2 = (np.array(Y - X10ia @ coef10ia_1[2:]) 
              @ ehat10ia_22.T) @ np.linalg.inv(pop_med_int.T @ (ehat10ia_22).T) 


#       b) D = search intensity

#1st stage--same as regression 8
coef10ib_1 = lasso8b.coef_.copy()

#2nd stage
X10ib = np.concatenate((agency_dummies,yrmon_dummies, int_controls_scaled), axis = 1)

#fit on uber_pen
lasso10ib_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=100000).fit(X10ib, uber_pen)
coef10ib_21 = lasso10ib_21.coef_
ehat10ib_21 = uber_pen.T- coef10ib_21.T @ X10ib.T

#fit on pop_med_int_pen
lasso10ib_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=100000).fit(X10ib, pop_med_int_pen)
coef10ib_22 = lasso10ib_22.coef_
ehat10ib_22 = pop_med_int_pen.T- coef10ib_22.T @ X10ib.T

#Calculate alpha
alpha10ib_B1 = (np.array(Y - X10ib @ coef10ib_1[2:]) 
              @ ehat10ib_21.T) @ np.linalg.inv(uber_pen.T @ (ehat10ib_21).T) 

alpha10ib_B2 = (np.array(Y - X10ib @ coef10ib_1[2:]) 
              @ ehat10ib_22.T) @ np.linalg.inv(pop_med_int_pen.T @ (ehat10ib_22).T) 




pop_poly_dblasso = pd.DataFrame([[alpha10ia_B1, alpha10ia_B2, alpha10ib_B1, alpha10ib_B2]],
                         columns=['dblasso_Uber_dummy','dblasso_Above_med*Uber_dummy',
                                   'dblasso_Uber_pen', 'dblasso_Above_med*Uber_pen'],
                          index=['coef'])

pop_poly_dblasso


# In[15]:


#  Regresion 10.2 --- For rides (corresponding to regression 9)

#       a) D= dummy
#1st stage--same as regression 9 
coef10iia_1 = lasso9a.coef_.copy()

#2nd stage
X10iia = np.concatenate((agency_dummies,yrmon_dummies,int_controls_scaled), axis = 1)

#fit on uber_dummy 
lasso10iia_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=500000).fit(X10iia, uber_dummy)
coef10iia_21 = lasso10iia_21.coef_
ehat10iia_21 = uber_dummy.T- coef10iia_21.T @ X10iia.T

#fit on rides_med_int
lasso10iia_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=500000).fit(X10iia, rides_med_int)
coef10iia_22 = lasso10iia_22.coef_
ehat10iia_22 = pop_med_int.T- coef10iia_22.T @ X10iia.T

#Calculate alpha
alpha10iia_B1 = (np.array(Y - X10iia @ coef10iia_1[2:]) 
              @ ehat10iia_21.T) @ np.linalg.inv(uber_dummy.T @ (ehat10iia_21).T) 

alpha10iia_B2 = (np.array(Y - X10iia @ coef10iia_1[2:]) 
              @ ehat10iia_22.T) @ np.linalg.inv(rides_med_int.T @ (ehat10iia_22).T) 


#       b) D = search intensity

#1st stage--same as regression 8
coef10iib_1 = lasso9b.coef_.copy()

#2nd stage
X10iib = np.concatenate((agency_dummies,yrmon_dummies, int_controls_scaled), axis = 1)

#fit on uber_pen
lasso10iib_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=500000).fit(X10iib, uber_pen)
coef10iib_21 = lasso10iib_21.coef_
ehat10iib_21 = uber_pen.T- coef10iib_21.T @ X10iib.T

#fit on rides_med_int_pen
lasso10iib_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                      max_iter=500000).fit(X10iib, rides_med_int_pen)
coef10iib_22 = lasso10iib_22.coef_
ehat10iib_22 = rides_med_int_pen.T- coef10iib_22.T @ X10iib.T

#Calculate alpha
alpha10iib_B1 = (np.array(Y - X10iib @ coef10iib_1[2:]) 
              @ ehat10iib_21.T) @ np.linalg.inv(uber_pen.T @ (ehat10iib_21).T) 

alpha10iib_B2 = (np.array(Y - X10iib @ coef10iib_1[2:]) 
              @ ehat10iib_22.T) @ np.linalg.inv(rides_med_int_pen.T @ (ehat10iib_22).T) 




rides_poly_dblasso = pd.DataFrame([[alpha10iia_B1, alpha10iia_B2, alpha10iib_B1, alpha10iib_B2]],
                         columns=['dblasso_Uber_dummy','dblasso_Above_med*Uber_dummy',
                                   'dblasso_Uber_pen', 'dblasso_Above_med*Uber_pen'],
                          index=['coef'])

rides_poly_dblasso


# In[ ]:





# In[19]:


# Regression 1&2 (OLS)
ols_model = pd.DataFrame([[betahat_1a[1],sdhat_1a,
                           "("+str(round(cil_1a,4))+","+str(round(cir_1a,4))+")",
                           betahat_1b[1],sdhat_1b,
                           "("+str(round(cil_1b,4))+","+str(round(cir_1b,4))+")"],
                          [betahat_2a[0],sdhat_2a,
                           "("+str(round(cil_2a,4))+","+str(round(cir_2a,4))+")",
                           betahat_2b[0],sdhat_2b,
                           "("+str(round(cil_2b,4))+","+str(round(cir_2b,4))+")"]], 
                          columns=['D=dummy_Coef','D=dummy_SE','D=dummy_95%CI',
                                   'D=pen_Coef','D=pen_SE','D=pen_95%CI'],
                         index=['OLS 1','OLS 2'])

ols_model


# In[28]:


#Regression 3-10 
# 3-4: OLS, adding pop & rides
# 5-6: Lasso, same as 3-4
# 7: Double Lasso on 5&6
# 8-9: Lasso, add poly
# 10: Double Lasso on 8&9

model_group = pd.DataFrame([
    [str(round(betahat_3a[0],4))+"("+str(round(sdhat_3a,4))+")",
     str(round(betahat_3a[1],4))+"("+str(round(sdhat_3a_pop,4))+")",
     str(round(betahat_3b[0],4))+"("+str(round(sdhat_3b,4))+")",
     str(round(betahat_3b[1],4))+"("+str(round(sdhat_3b_pop,4))+")"],
    [str(round(betahat_4a[0],4))+"("+str(round(sdhat_4a,4))+")",
     str(round(betahat_4a[1],4))+"("+str(round(sdhat_4a_rides,4))+")",
     str(round(betahat_4b[0],4))+"("+str(round(sdhat_4b,4))+")",
     str(round(betahat_4b[1],4))+"("+str(round(sdhat_4b_rides,4))+")"],
    [round(coef5a[0],4),round(coef5a[1],4),round(coef5b[0],4),round(coef5b[1],4)],
    [round(coef6a[0],4),round(coef6a[1],4),round(coef6b[0],4),round(coef6b[1],4)],
    [np.round(alpha7ia_B1,4),np.round(alpha7ia_B2,4),np.round(alpha7ib_B1,4),
     np.round(alpha7ib_B2,4)],
    [np.round(alpha7iia_B1,4),np.round(alpha7iia_B2,4),np.round(alpha7iib_B1,4),
     np.round(alpha7iib_B2,4)],
    [round(coef8a[0],4),round(coef8a[1],4),round(coef8b[0],4),round(coef8b[1],4)],
    [round(coef9a[0],4),round(coef9a[1],4),round(coef9b[0],4),round(coef9b[1],4)],
    [np.round(alpha10ia_B1,4),np.round(alpha10ia_B2,4),np.round(alpha10ib_B1,4),
     np.round(alpha10ib_B2,4)],
    [np.round(alpha10iia_B1,4),np.round(alpha10iia_B2,4),np.round(alpha10iib_B1,4),
     np.round(alpha10iib_B2,4)]],
    columns=['D=dummy','D=dummy*Above_med','D=pen','D=pen*Above_med'],
    index=['OLS3_pop','OLS4_rides','Lasso_pop','Lasso_rides','DbLasso_pop',
           'DbLasso_rides','Lasso_poly_pop','Lasso_poly_rides',
           'DbLasso_poly_pop','DbLasso_poly_rides'])

model_group


# In[ ]:





# In[ ]:





# In[ ]:


###################### BONUS REGRESSION 1-6 #####################

#New D = dummy * search intensity

#This will be equal to zero before Uber enters (when dummy=0), and equal to 
#the search intensity volume after. This combines the effects of Ubers presence
#from the dummy variable with the size/populariy of uber from the search intensity

#Bonus 1 - Reg 3 but with new D

data["D_new"] = data["treatUberX"] * data["treatGTNotStd"]
D_new = np.array(data["D_new"], ndmin = 2).T

data["pop_int_new"] = data["pop_med_dummy"] * data["D_new"]
pop_int_new = np.array(data["pop_int_new"], ndmin = 2).T

#And we will fit into different regressions to see if it improve the result.


Xbonus1_1 = np.concatenate((D_new, pop_int_new, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_b1 = np.linalg.inv(Xbonus1_1.T @ Xbonus1_1) @ (Xbonus1_1.T @ Y)
ehat_b1 = Y - Xbonus1_1 @ betahat_b1
ehat_b1 = np.array(ehat_b1, ndmin = 2).T

Sigmahat_b1 = (Xbonus1_1 * ehat_b1).T @ (Xbonus1_1 * ehat_b1) / n

Qhat_b1 = np.linalg.inv(Xbonus1_1.T @ Xbonus1_1 / n)
Vhat_b1 = Qhat_b1 @ Sigmahat_b1 @ Qhat_b1
sdhat_b1 = np.sqrt(Vhat_b1[0 ,0]) / np.sqrt(n)

sdhat_b1_pop = np.sqrt(Vhat_b1[1 ,1]) / np.sqrt(n)


#Bonus 2 - Reg 4 but with new D

data["rides_int_new"] = data["rides_med_dummy"] * data["D_new"]
rides_int_new = np.array(data["rides_int_new"], ndmin = 2).T

Xbonus_2 = np.concatenate((D_new, rides_int_new, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)

betahat_b2 = np.linalg.inv(Xbonus_2.T @ Xbonus_2) @ (Xbonus_2.T @ Y)
ehat_b2 = Y - Xbonus_2 @ betahat_b2
ehat_b2 = np.array(ehat_b2, ndmin = 2).T

Sigmahat_b2 = (Xbonus_2 * ehat_b2).T @ (Xbonus_2 * ehat_b2) / n

Qhat_b2 = np.linalg.inv(Xbonus_2.T @ Xbonus_2 / n)
Vhat_b2 = Qhat_b2 @ Sigmahat_b2 @ Qhat_b2
sdhat_b2 = np.sqrt(Vhat_b2[0 ,0]) / np.sqrt(n)

sdhat_b2_rides = np.sqrt(Vhat_b2[1 ,1]) / np.sqrt(n)
#Vhat_b2[1 ,1] is negative ????

#Bonus 3 - Reg 5 with new D

Xbonus_3 = np.concatenate((D_new, pop_int_new, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)


#run lasso
lassob3 = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lassob3.fit(Xbonus_3 ,Y)
coefb3 = lassob3.coef_


#Bonus 4 - Reg 6 with new D

Xbonus_4 = np.concatenate((D_new, rides_int_new, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)


#run lasso
lassob4 = LassoCV(cv = 5, fit_intercept=False,  random_state=0)
lassob4.fit(Xbonus_4 ,Y)
coefb4 = lassob4.coef_


# In[59]:


#Bonus 5 - Reg 8 with new D

Xbonus_5 = np.concatenate((D_new, pop_int_new, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)

lassob5 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob5.fit(Xbonus_5 ,Y)
coefb5 = lassob5.coef_


#Bonus 6 - Reg 9 with new D

Xbonus_6 = np.concatenate((D_new, rides_int_new, agency_dummies, 
                      yrmon_dummies, int_controls_scaled), axis = 1)

lassob6 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob6.fit(Xbonus_6 ,Y)
coefb6 = lassob6.coef_


# In[60]:


#Bonus Regression1-6: 

bonus_group1 = pd.DataFrame([
    [str(round(betahat_b1[0],4))+"("+str(round(sdhat_b1,4))+")",
     str(round(betahat_b1[1],4))+"("+str(round(sdhat_b1_pop,4))+")"],
    [str(round(betahat_b2[0],4))+"("+str(round(sdhat_b2,4))+")",
     str(round(betahat_b2[1],4))+"("+str(round(sdhat_b2_rides,4))+")"],
    [round(coefb3[0],4),round(coefb3[1],4)],[round(coefb4[0],4),round(coefb4[1],4)],
    [round(coefb5[0],4),round(coefb5[1],4)],[round(coefb6[0],4),round(coefb6[1],4)]],
    columns=['D_new','D_new*Above_med'], 
    index=['Bonus1_pop_OLS','Bonus2_rides_OLS','Bonus3_pop_Lasso','Bonus4_rides_Lasso',
           'Bonus5_pop_LassoPoly','Bonus6_rides_LassoPoly'])

bonus_group1


# In[39]:


###################### BONUS REGRESSION 7- #####################

#put the interaction of pop*dummy and rides*dummy in one model as the paper does and 
#fit OLS, Lasso rgeression

#Bonus 7 - OLS but with both inteactions.

# a) with D=dummy
Xbonus7_1 = np.concatenate((uber_dummy, pop_med_int, rides_med_int, agency_dummies, 
                      yrmon_dummies, controls), axis = 1)


# In[ ]:


betahat_b7_1 = np.linalg.inv(Xbonus7_1.T @ Xbonus7_1) @ (Xbonus7_1.T @ Y)
ehat_b7_1 = Y - Xbonus7_1 @ betahat_b7_1
ehat_b7_1 = np.array(ehat_b7_1, ndmin = 2).T

Sigmahat_b7_1 = (Xbonus7_1 * ehat_b7_1).T @ (Xbonus7_1 * ehat_b7_1) / n

Qhat_b7_1 = np.linalg.inv(Xbonus7_1.T @ Xbonus7_1 / n)
Vhat_b7_1 = Qhat_b7_1 @ Sigmahat_b7_1 @ Qhat_b7_1
sdhat_b7_1 = np.sqrt(Vhat_b7_1[0 ,0]) / np.sqrt(n)

sdhat_b7_1pop = np.sqrt(Vhat_b7_1[1 ,1]) / np.sqrt(n)
sdhat_b7_1rides = np.sqrt(Vhat_b7_1[2 ,2]) / np.sqrt(n)


# In[43]:


# b) with D=search intensity
Xbonus7_2 = np.concatenate((uber_pen, pop_med_int_pen, rides_med_int_pen, 
                            agency_dummies, yrmon_dummies, controls), axis = 1)


# In[47]:


betahat_b7_2 = np.linalg.inv(Xbonus7_2.T @ Xbonus7_2) @ (Xbonus7_2.T @ Y)
ehat_b7_2 = Y - Xbonus7_2 @ betahat_b7_2
ehat_b7_2 = np.array(ehat_b7_2, ndmin = 2).T

Sigmahat_b7_2 = (Xbonus7_2 * ehat_b7_2).T @ (Xbonus7_2 * ehat_b7_2) / n

Qhat_b7_2 = np.linalg.inv(Xbonus7_2.T @ Xbonus7_2 / n)
Vhat_b7_2 = Qhat_b7_2 @ Sigmahat_b7_2 @ Qhat_b7_2
sdhat_b7_2 = np.sqrt(Vhat_b7_2[0 ,0]) / np.sqrt(n)

sdhat_b7_2pop = np.sqrt(Vhat_b7_2[1 ,1]) / np.sqrt(n)
sdhat_b7_2rides = np.sqrt(Vhat_b7_2[2 ,2]) / np.sqrt(n)


# In[40]:


#Bonus 8 - Lasso but with both inteactions.

# a) with D=dummy
Xbonus8_1 = Xbonus7_1.copy()


# In[46]:


lassob8_1 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob8_1.fit(Xbonus8_1 ,Y)
coefb8_1 = lassob8_1.coef_


# In[47]:


# b) with D=search intensity
Xbonus8_2 = Xbonus7_2.copy()

lassob8_2 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob8_2.fit(Xbonus8_2 ,Y)
coefb8_2 = lassob8_2.coef_


# In[49]:


#Bonus 9 - Lasso of Poly with both inteactions.

# a) with D=dummy
Xbonus9_1 =  np.concatenate((uber_dummy, pop_med_int, rides_med_int, agency_dummies, 
                      yrmon_dummies, controls_scaled), axis = 1)


lassob9_1 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob9_1.fit(Xbonus9_1 ,Y)
coefb9_1 = lassob9_1.coef_

# b) with D=search intensity
Xbonus9_2 =  np.concatenate((uber_pen, pop_med_int_pen, rides_med_int_pen, 
                             agency_dummies, yrmon_dummies, controls_scaled), axis = 1)

lassob9_2 = LassoCV(cv = 5, fit_intercept=False, max_iter=100000, random_state=0)
lassob9_2.fit(Xbonus9_2 ,Y)
coefb9_2 = lassob9_2.coef_


# In[52]:


#Bonus 10 - DoubleLasso with both inteactions, corresponding to 8

# a) with D=dummy
#1st stage--same as Bonus 8_1 
coefb10_1 = coefb8_1.copy()

#2nd stage
Xbonus10_1 = np.concatenate((agency_dummies, yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_dummy 
lassob10_11 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_1, uber_dummy)
coefb10_11 = lassob10_11.coef_
ehatb10_11 = uber_dummy.T- coefb10_11.T @ Xbonus10_1.T

#fit on pop_med_int
lassob10_12 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_1, pop_med_int)
coefb10_12 = lassob10_12.coef_
ehatb10_12 = pop_med_int.T- coefb10_12.T @ Xbonus10_1.T

#fit on rides_med_int
lassob10_13 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_1, rides_med_int)
coefb10_13 = lassob10_13.coef_
ehatb10_13 = pop_med_int.T- coefb10_13.T @ Xbonus10_1.T


#Calculate alpha
alphab10_11 = (np.array(Y - Xbonus10_1 @ coefb10_1[3:]) 
              @ ehatb10_11.T) @ np.linalg.inv(uber_dummy.T @ (ehatb10_11).T) 

alphab10_12 = (np.array(Y - Xbonus10_1 @ coefb10_1[3:]) 
              @ ehatb10_12.T) @ np.linalg.inv(pop_med_int.T @ (ehatb10_12).T)

alphab10_13 = (np.array(Y - Xbonus10_1 @ coefb10_1[3:]) 
              @ ehatb10_13.T) @ np.linalg.inv(rides_med_int.T @ (ehatb10_13).T)

# b) with D=search intensity
#1st stage--same as Bonus 8_2
coefb10_2 = coefb8_2.copy()

#2nd stage
Xbonus10_2 = np.concatenate((agency_dummies, yrmon_dummies, controls_scaled), axis = 1)

#fit on uber_dummy 
lassob10_21 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_2, uber_pen)
coefb10_21 = lassob10_21.coef_
ehatb10_21 = uber_dummy.T- coefb10_21.T @ Xbonus10_2.T

#fit on pop_med_int
lassob10_22 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_2, pop_med_int_pen)
coefb10_22 = lassob10_22.coef_
ehatb10_22 = pop_med_int_pen.T- coefb10_22.T @ Xbonus10_2.T

#fit on rides_med_int
lassob10_23 = LassoCV(cv = 5, fit_intercept=False,random_state=0, 
                       max_iter=10000).fit(Xbonus10_2, rides_med_int_pen)
coefb10_23 = lassob10_23.coef_
ehatb10_23 = pop_med_int_pen.T- coefb10_23.T @ Xbonus10_2.T


#Calculate alpha
alphab10_21 = (np.array(Y - Xbonus10_2 @ coefb10_2[3:]) 
              @ ehatb10_21.T) @ np.linalg.inv(uber_dummy.T @ (ehatb10_21).T) 

alphab10_22 = (np.array(Y - Xbonus10_2 @ coefb10_2[3:]) 
              @ ehatb10_22.T) @ np.linalg.inv(pop_med_int_pen.T @ (ehatb10_22).T)

alphab10_23 = (np.array(Y - Xbonus10_2 @ coefb10_2[3:]) 
              @ ehatb10_23.T) @ np.linalg.inv(rides_med_int_pen.T @ (ehatb10_23).T)


# In[57]:


#Bonus Regression: 

bonus_group2 = pd.DataFrame([
    [str(round(betahat_b7_1[0],4))+"("+str(round(sdhat_b7_1,4))+")",
     str(round(betahat_b7_1[1],4))+"("+str(round(sdhat_b7_1pop,4))+")",
     str(round(betahat_b7_1[2],4))+"("+str(round(sdhat_b7_1rides,4))+")"],
    [str(round(betahat_b7_2[0],4))+"("+str(round(sdhat_b7_2,4))+")",
     str(round(betahat_b7_2[1],4))+"("+str(round(sdhat_b7_2pop,4))+")",
     str(round(betahat_b7_2[2],4))+"("+str(round(sdhat_b7_2rides,4))+")"],
    [round(coefb8_1[0],4),round(coefb8_1[1],4),round(coefb8_1[2],4)],
    [round(coefb8_2[0],4),round(coefb8_2[1],4),round(coefb8_2[2],4)],
    [round(coefb9_1[0],4),round(coefb9_1[1],4),round(coefb9_1[2],4)],
    [round(coefb9_2[0],4),round(coefb9_2[1],4),round(coefb9_2[2],4)],
    [np.round(alphab10_11,4),np.round(alphab10_12,4),np.round(alphab10_13,4)],
    [np.round(alphab10_21,4),np.round(alphab10_22,4),np.round(alphab10_23,4)]],
    columns=['D','D*Above_med_pop','D*Above_med_rides'], 
    index=['Bonus7_dummy_OLS','Bonus7_pen_OLS','Bonus8_dummy_Lasso','Bonus8_pen_Lasso',
          'Bonus9_dummy_LassoPoly','Bonus9_pen_LassoPoly','Bonus10_dummy_DbLasso',
           'Bonus10_pen_DbLasso'])

bonus_group2

