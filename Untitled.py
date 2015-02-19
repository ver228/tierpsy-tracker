# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""

import sqlite3 as sql

fileName = '/Volumes/ajaver$/Video/Exp4-20141216/A001 - 20141216_195148.wmv';
DBName = '/Volumes/ajaver$/Video/Exp4-20141216/A001_results.db';


conn = sql.connect(DBName)

conn.close()
