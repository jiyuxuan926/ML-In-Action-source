# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from sklearn import linear_model

reg = linear_model.Lasso(alpha=0.1)

reg.fit([[0,0],[1,1],[2,2]],[0,1,2])

a= reg.coef_   # 使用coef_关键字得到的是线性模型的系数

print(a)


b= reg.intercept_ 

print(b)

print(reg.predict([[1,1]]))


