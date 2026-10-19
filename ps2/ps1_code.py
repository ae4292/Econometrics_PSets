#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# %%
wages = pd.read_csv("ps1small.csv")
#%%
wages['logwage'] = np.log(wages['wage'])
wages['eduxage'] = 'd'+ wages['education'].astype(str) + wages['age'].astype(str)
dummies = pd.get_dummies(wages['eduxage'], dtype = int)
wages = pd.concat([wages, dummies], axis=1)
#%%
wages.head()
#%%# %%
model = sm.OLS(endog= wages['logwage'], exog=wages[dummies.columns]).fit()
y_hat = model.predict(wages.iloc[:, 5:])
coefficients = model.params
standard_errors = model.bse
alpha = pd.DataFrame({'Coefficient': coefficients, 'Standard Error': standard_errors})
alpha
# the combined effect of age and education on logwage are not statistically different from one another.

# %%
alpha_vec = alpha['Coefficient']

R  = np.zeros((26,30))
q = np.zeros((26,1))

col = lambda e, a: (e - 12) * 5 + (a - 26)   # column of alpha_{e,a}

r = 0
# Block 1 (18 rows): linear in age within each education level
for e in range(12, 18):
    for a in (27, 28, 29):
        R[r, col(e, a-1)] += 1
        R[r, col(e, a)]   -= 2
        R[r, col(e, a+1)] += 1
        r += 1

# Block 2 (4 rows): age slope s_e = alpha_{e,27} - alpha_{e,26} is linear in education
for e in range(13, 17):
    for w, ee in ((1, e-1), (-2, e), (1, e+1)):
        R[r, col(ee, 27)] += w
        R[r, col(ee, 26)] -= w
    r += 1

# Block 3 (4 rows): level at age 26 is linear in education
for e in range(13, 17):
    R[r, col(e-1, 26)] += 1
    R[r, col(e, 26)]   -= 2
    R[r, col(e+1, 26)] += 1
    r += 1

#%%
wald_test = model.wald_test((R,q), use_f = True)
F = wald_test.fvalue
p_value = wald_test.pvalue
print("F-statistic:", F)
print("p-value:", p_value)
# %%
