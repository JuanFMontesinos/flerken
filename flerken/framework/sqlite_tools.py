#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3

def connect(path, verbose = False):
	conn = sqlite3.connect(path)
	cursor = conn.cursor()
	if verbose:
		print('Opened database at {} sucessfully'.format(path))
	return conn,cursor

def init_table(cursor,table,verbose = False):
	cursor.execute('''CREATE TABLE {0}
         (ID TEXT PRIMARY KEY     NOT NULL,
         DATE_OF_CREATION            TEXT     NOT NULL,
         MODEL TEXT ,
         LR REAL,
         LOSS REAL,
         ACC REAL,
         VLOSS REAL,
         VACC REAL,
         EPOCH INTEGER,
         OPTIMIZER TEXT,
         CHECKPOINT INTEGER,
         ITERATIONS INTEGER);'''.format(table))
	if verbose:
		print('Table sucessfully created')

def names2str(value_names):
	text = str()
	for f in value_names:
		text += f + ','
	return text[:-1]
def fields2str(value_names):
	text = str()
	for f in value_names:
		if type(f) == str:
			text += "'"+f+"'" + ','
		else:
			text += str(f) + ','
	return text[:-1]
def values2update(value_names,value_fields):
	value_fields = [q if type(q) != str else "'"+q+"'" for q in value_fields ]
	string = str()
	for n,f in zip(value_names,value_fields):
		string+= "{0} = {1},".format(n,f)
	return string[:-1]
def insert_value(cursor,conn,values,table):
	value_names = list(values.keys())
	value_field = [values[key] for key in value_names]
	cursor.execute("INSERT INTO {0} ({1}) \
      VALUES ({2})".format(table,names2str(value_names),fields2str(value_field)))
	conn.commit()

def delete(cursor,conn,id,table):
	id = "'"+id+"'"
	cursor.execute("DELETE from  {0} where ID = {1}".format(table,id))
	conn.commit()
def exists(cursor,id,table,rowid):
	id = "'"+id+"'"
	cursor.execute("SELECT {0} FROM {1} WHERE {0} = {2}".format(rowid,table,id))
	if cursor.fetchone() == None:
		return False
	else:
		return True
def update(cursor,conn,values,table):
	value_names = list(values.keys())
	value_field = [values[key] for key in value_names]
	rowid = value_names[0]
	idx = "'"+value_field[0]+"'"
	del value_names[0]
	del value_field[0]
	vals=values2update(value_names,value_field)

	cursor.execute("UPDATE {0} SET {1} WHERE {2} = {3}".format(table,vals,rowid,idx))
	conn.commit()
class sq(object):
	def __init__(self,path):
		self.dst = path
		self.conn,self.cursor = connect(path)
		self.table = 'EXPERIMENTS'
		try:
			init_table(self.cursor,self.table)
		except:
			pass

	def insert_value(self,values):
		insert_value(self.cursor,self.conn,values,self.table)
	def delete(self,id):
		delete(self.cursor,self.conn,id,self.table)
	def destructor(self):
		self.cursor.close()
		self.conn.close()
	def exists(self,id):
		return exists(self.cursor,id,self.table,'ID')
	def update(self,values):
		update(self.cursor,self.conn,values,self.table)
def test():
	db = sq('test_db.sqlite')
	tester = {'ID':'kawasaki','DATE_OF_CREATION':151}
	tester2 = {'ID':'nagasaki','DATE_OF_CREATION':115}
	tester3 = {'ID':'nagasaki','DATE_OF_CREATION':55,'LR':0.01}
	db.insert_value(tester)
	db.insert_value(tester2)
	db.delete('kawasaki')
	print(db.exists('kawasaki'))
	print(db.exists('nagasaki'))
	db.update(tester3)
	print('test sucessful')

