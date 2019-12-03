import numpy as np
import math
import probclearn
import probcpredict
# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(k,X,y):
            # Your code goes here
    n = len(X)
    z = np.zeros(k)
    for i in range(k):
        T = []
        for j in range(int(math.floor((n*i)/k)), int(math.floor((n*(i+1))/(k)))):
            T.append(j)

            S = set((range(n)))-set(T)
            S = np.array(list(S))

            Xtrain = []
            for t in S:
                Xtrain.append(X[t])

                ytrain = []
                for t in S:
                    ytrain.append(y[t])    

                q,mu_pos,mu_neg,sigma2_pos,sigma2_neg = probclearn.run(Xtrain,ytrain)    

                for t in T:
                    temp = np.array([X[t]]).T
                    if y[t] != probcpredict.run(q,mu_pos,mu_neg,sigma2_pos,sigma2_neg,temp):
                        z[i] = z[i]+1
                z[i] = z[i]/len(T)    
    return np.array([z]).T

