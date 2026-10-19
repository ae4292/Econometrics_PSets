#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# %%
wages = pd.read_csv("ps1small.csv")

wages['logwage'] = np.log(wages['wage'])
wages['eduxage'] = 'd'+ wages['education'].astype(str) + wages['age'].astype(str)
dummies = pd.get_dummies(wages['eduxage'], dtype = int)
wages = pd.concat([wages, dummies], axis=1)
wages.head()
#%%
model = sm.OLS(endog= wages['logwage'], exog=wages[dummies.columns]).fit()
coefficients = model.params
standard_errors = model.bse
alpha = pd.DataFrame({'Coefficient': coefficients, 'Standard Error': standard_errors})
alpha
#%%
# alpha is the individual effect of having a given age and education on log wage.
#%%

R  = np.zeros((26,30))
q = np.zeros((26,1))

def col(e, a):
    # column index for education level e and age a
    return (e - 12) * 5 + (a - 26)   

#alpha_{e,a-1} - 2*alpha_{e,a} + alpha_{e,a+1} = 0
r = 0
for e in range(12, 18):
    for a in (27, 28, 29):
        R[r, col(e, a-1)] = 1
        R[r, col(e, a)]   = -2
        R[r, col(e, a+1)] = 1
        r += 1

#(alpha_{e-1,27} - alpha_{e-1,26}) - 2*(alpha_{e,27} - alpha_{e,26}) + (alpha_{e+1,27} - alpha_{e+1,26}) = 0
for r in range(18, 22):
    e = 13 + (r - 18)     
    R[r, col(e-1, 27)] = 1
    R[r, col(e-1, 26)] = -1
    R[r, col(e, 27)]   = -2
    R[r, col(e, 26)]   = 2
    R[r, col(e+1, 27)] = 1
    R[r, col(e+1, 26)] = -1

#alpha_{e-1,26} - 2*alpha_{e,26} + alpha_{e+1,26} = 0
for r in range(22, 26):
    e = 13 + (r - 22)
    R[r, col(e-1, 26)] = 1
    R[r, col(e, 26)]   = -2
    R[r, col(e+1, 26)] = 1

#%%
wald_test = model.wald_test((R,q), use_f = True)
F = wald_test.fvalue
p_value = wald_test.pvalue
print("F-statistic:", F)
print("p-value:", p_value)
# %%
# I woul not reject the null hypothesis as the p-value is not below 0.05 significance level.