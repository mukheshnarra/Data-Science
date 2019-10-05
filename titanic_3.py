# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:56:56 2018

@author: MUKHESH
"""

import pandas as pd
data=pd.read_csv("C:/Users/MUKHESH/Documents/PYTHON/titanic.csv")
data.loc[(data.Name.str.contains('Mrs')),'Sex']="female"
data.loc[(data.Name.str.contains('Miss')),'Sex']="female"
data.loc[(data.Name.str.contains('Master')),'Sex']="male"
data.loc[data['Sex'].isnull(),'Sex']="male"
data.to_csv("C:/Users/MUKHESH/Documents/PYTHON/titanic_modified.csv",index=False)