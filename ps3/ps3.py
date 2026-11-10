#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
# %%
data = pd.read_excel("PS4data.xls")
data['consumption'] = data['real consumption of nondurables'] + \
data['real consumption of services']
data['consumption per capita'] = data['consumption'] / data['population']  
data['consumption per capita'] 
# %%
lags= data['consumption per capita'].shift(periods=[1,2,3,4])
data_with_lags = pd.concat([data, lags], axis=1)
# %%
data_with_lags_1 = data_with_lags.iloc[4:, :]

model = sm.OLS(data_with_lags_1['consumption per capita'], sm.add_constant(data_with_lags_1.iloc[:, -4:]))

model_fit = model.fit()

r_matrix = np.identity(5)[2:, :]
print(model_fit.f_test(r_matrix = r_matrix).pvalue)
# %%
# We reject Hall's Hypothesis that beta_2 = beta_3 = beta_4 = 0 at the 5% significance level since the p-value is less than 0.05.
#%%
## Part 2b
#%%
#This setup from Campbell and Mankiw has an endogenity issue since current income is a part of permanent income. Thus the remainder is correlated with the error term. This will lead to inconsistent estimate of gamma_1.
#%%
## Part 2c
#%%
from linearmodels.iv import IV2SLS
for i in range(0, 6):
    data_with_lags[f'log change consumption per capita_lag{i}'] = np.log(data_with_lags['consumption per capita'].shift(i) / data_with_lags['consumption per capita'].shift(i + 1))
endog = data_with_lags['log change consumption per capita_lag0']

data_with_lags['real disposable income per capita'] = data_with_lags['real disposable income'] / data_with_lags['population']
data_with_lags['real disposable income per capita_1'] = data_with_lags['real disposable income per capita'].shift(periods=1)
exog = sm.add_constant(np.log(data_with_lags['real disposable income per capita'] / data_with_lags['real disposable income per capita_1']))

instruments = data_with_lags[[f'log change consumption per capita_lag{i}' for i in range(2, 6)]]

iv_model = IV2SLS(endog[6:], exog[['const']][6:], exog.iloc[:, 1][6:], instruments[6:])
iv_results = iv_model.fit(cov_type='robust')
print(iv_results.params)
print(iv_results.std_errors)
print(iv_results.pvalues)
# %%
# gamma_0 is small, but statisitically significant. Gamma_1 is 0.436 and statistically significant. Thus, with this 2SLS regression the estimated fraction of consumers that spend current income is 0.436.

#%%
## Part 2d
#%%
print(f'P-value of exogeneity test: {iv_results.wooldridge_regression.pval}')
# %%
n = len(data_with_lags[6:])
from scipy.stats import chi2

W = np.column_stack([np.ones(n), instruments[6:].values])
error = iv_results.resids.values.reshape(-1, 1)

S = n * (error.T @ W @ np.linalg.inv(W.T @ W) @ W.T @ error) / (error.T @ error)
chi2_stat = S[0][0]

l, k = W.shape[1], 2
p_value = 1 - chi2.cdf(chi2_stat, df=l - k)

print(f'Sargan statistic: {chi2_stat}, p-value: {p_value}')
# %%
# Since the exogeneity test p-value is greater than 0.05, we fail to reject the null hypothesis that log change in real disposable income per capita is not exogenous. Since the Sargan test p-value is greater than 0.05, we fail to reject the null hypothesis that the instruments are not exogenous. We can conclude that the instruments are valid and the 2SLS results are consistent.