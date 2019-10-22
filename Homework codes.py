import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_excel('Industry_Portfolios.xlsx',index_col=0,parse_dates=False)

M = np.mean(data)
SD = np.std(data)
Cov = data.cov()

Cov_inverse = np.linalg.inv(Cov)
e = np.array([1 for i in range(10)])

#minimum variance frontier without riskless asset
alpha=M.dot(Cov_inverse).dot(e)
zeta = M.dot(Cov_inverse).dot(M)
delta=e.dot(Cov_inverse).dot(e)

#global minimum variance portfolio
Rmv= alpha/delta
Rp = np.arange(0.00,2.00,0.01)
Sigma= np.sqrt((1/delta)+(delta/(zeta*delta - alpha**2))*(Rp-Rmv)**2)

#first plot
plt.figure()
plt.ylabel('Mean Return')
plt.xlabel('Standard Deviation')
plt.title('Minimum Variance Frontier without riskless asset')
plt.plot(Sigma,Rp, 'b-')

#risk free
Rf=0.13
Rp_1= np.arange(Rf,2.00,0.01)
Sigma_1= np.sqrt((Rp_1-Rf)**2/(zeta-2*alpha*Rf+delta*Rf**2))

#second plot
plt.figure()
plt.ylabel('Mean Return')
plt.xlabel('Standard Deviation')
plt.title('Minimum Variance Frontier')
plt.plot(Sigma_1,Rp_1, 'r-', label= 'with riskless asset')
plt.plot(Sigma,Rp, 'b-',label= 'without riskless asset')

#At Tangency Portfolio
R_tg=Rmv-((zeta*delta-alpha**2)/((delta**2)*(Rf-Rmv)))
Sigma_2=np.sqrt((zeta-2*alpha*Rf+delta*Rf**2)/((delta**2)*(Rf-Rmv)**2))
a=(((zeta*(Cov_inverse).dot(e))-(alpha*(Cov_inverse).dot(M)))/(zeta*delta-alpha**2))
b=((delta*Cov_inverse.dot(M))-(alpha*Cov_inverse.dot(e)))/(zeta*delta-alpha**2)
Weights = a+b*R_tg
Weights


#Table showing mean and standard deviation

d = {'Mean Return %':M, 'Standard Deviation %':SD}
data_1= pd.DataFrame(data=d)
print(data_1)
print(Cov)
print(Weights)

data_1.to_excel(r'C:\Users\ACER-PC\Desktop\asset pricing homework\datatable.xlsx',index=None,header=True)
