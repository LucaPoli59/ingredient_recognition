import os
import pandas as pd


class EasyPandas(pd.DataFrame):
	def __init__(self, *args, **kwargs):
		# get attributes that we need
		index_columns = kwargs.get('index_columns')
		secondary_columns = kwargs.get('secondary_columns', None)
		filepath = kwargs.get('filepath', 'save_pd.csv')
		continuous_saving = kwargs.get('continuous_saving', True)
		# remove keys from kwargs
		kwargs.pop('index_columns', None)
		kwargs.pop('secondary_columns', None)
		kwargs.pop('filepath', None)
		kwargs.pop('continuous_saving', None)
		# initialize pandas dataframe
		super().__init__(args, kwargs)
		# save attributes
		self.index_columns = index_columns
		self.secondary_columns = secondary_columns
		self.filepath = filepath
		self.continuous_saving = continuous_saving
		# initialize object
		if not self.restore():
			# if the dataframe does not exist and columns were not specified, throw an error
			if not self.secondary_columns:
				raise ValueError('Columns must be specified if the dataframe does not exist.')
			# define columns
			all_columns = list(set(self.index_columns + self.secondary_columns))
			# create empty dataframe
			empty_df = pd.DataFrame(columns=all_columns)
			self.__dict__.update(empty_df.__dict__)
			# save it
			self.save()

	
	def save(self):
		'''Saves the dataframe to a file using pandas.'''
		self.to_csv(self.filepath, index=False)
		self.to_excel(self.filepath.replace('.csv', '.xlsx'), index=False)
		

	def restore(self):
		'''Restores the dataframe from a file using pandas.'''
		if os.path.exists(self.filepath):
			# load the dataframe
			new_df = pd.read_csv(self.filepath)
			self.__dict__.update(new_df.__dict__)
			return True
		else:
			return False
		

	def exists(self, index_values):
		'''Checks if the index values exist in the dataframe.'''
		#  convert the index values to a dictionary if they are not
		if not isinstance(index_values, dict):
			index_values = dict(zip(self.index_columns, index_values))
		# convert the index values to a dataframe
		cur_vals = pd.DataFrame([index_values])
		# check if the index values exist
		return self.set_index(self.index_columns).index.isin(cur_vals.set_index(self.index_columns).index).any()
	

	def get(self, index_values=None):
		'''Returns the dataframe (if index_values are None) or the row corresponding to the index values.
		If the index values do not exist, returns False.'''
		if index_values is None:
			# if no index values are given, return the whole dataframe
			return self
		else:
			# if index values are given, check if they exist
			if not self.exists(index_values):
				return False
			# convert index values to a dictionary if they are not
			if not isinstance(index_values, dict):
				index_values = dict(zip(self.index_columns, index_values))
			# convert index in a dataframe
			index_valuess = pd.DataFrame(index_values, index=[0])
			index_valuess.columns = self.index_columns
			# create mask to locate the row
			mask = self.set_index(self.index_columns).index.isin(index_valuess.set_index(self.index_columns).index)
			# return it as a dictionary
			return self[mask].iloc[0].to_dict()

		

	def add(self, row):
		'''Adds a row to the dataframe. If the index values already exist, the row is not added.'''
		# check type
		assert(isinstance(row, dict))
		# check if the index values exist
		# if self.exists(row[self.index_columns]):
		if self.exists(row):
			return False
		# convert the row to a dataframe
		cur_val = pd.DataFrame(row, index=[0])
		# add the row
		if self.empty:
			# if the dataframe is empty, just set the row as the dataframe
			self.__dict__.update(cur_val.__dict__)
		else:
			# if the dataframe is not empty, append the row to the dataframe
			new_df = pd.concat([self, cur_val], ignore_index=True)
			self.__dict__.update(new_df.__dict__)
		# save the dataframe
		if self.continuous_saving:
			self.save()
		return True
	

	def update(self, row):
		'''Updates a row in the dataframe. If the index values do not exist, the row is not updated.'''
		# check type
		assert(isinstance(row, dict))
		# check if the index values exist
		if not self.exists(row):
			return False
		# remove previous value
		cur_idx = pd.DataFrame(row, index=[0])[self.index_columns]
		new_df = self.set_index(self.index_columns).drop(cur_idx.set_index(self.index_columns).index)
		# add the row with new values
		new_df = pd.concat([new_df, pd.DataFrame(row, index=[0]).set_index(self.index_columns)])
		new_df.reset_index(drop=False, inplace=True)
		# sort by index columns
		new_df = new_df.sort_values(by=self.index_columns)
		# update the dataframe
		self.__dict__.update(new_df.__dict__)
		# save the dataframe
		if self.continuous_saving:
			self.save()
		# return True
		return True
	

	def delete(self, index_values):
		'''Deletes a row from the dataframe. If the index values do not exist, the row is not deleted.'''
		# check if the index values exist
		if not self.exists(index_values):
			return False
		# delete the row
		new_df = self.set_index(self.index_columns).drop(index_values)
		self.__dict__.update(new_df.__dict__)
		# save the dataframe
		if self.continuous_saving:
			self.save()
		return True





if __name__ == '__main__':
	import itertools
	import random
	import time

	# define parameters
	architecture = ['resnet18', 'resnet34', 'resnet50']
	learning_rate = [0.1, 0.01, 0.001]
	# define all possible combinations of the parameters that need to be explored with itertools
	experiments = list(itertools.product(architecture, learning_rate))

	# create object
	easy_df = EasyPandas(index_columns = ['architecture', 'learning_rate'], secondary_columns=['accuracy'], filepath='save.csv')


	# for every experiment
	for experiment in experiments:
		# check if the experiment exists
		if not easy_df.exists(experiment):
			# simulate experiment
			print('Simulating experiment with architecture {} and learning rate {}'.format(experiment[0], experiment[1]))
			time.sleep(0.5)
			# compute the accuracy of the experiment
			accuracy = round(random.random(),2)
			# add the experiment to the dataframe (it will save it automatically on disk)
			easy_df.add({'architecture': experiment[0], 'learning_rate': experiment[1], 'accuracy': accuracy})

	# let's print the final dataframe
	print(easy_df)

	# it is also possible to get rows
	print(easy_df.get(('resnet18', 0.1)))
	print(easy_df.get({'architecture':'resnet18', 'learning_rate':0.1}))

	# it is also possible to update rows
	easy_df.update({'architecture': 'resnet18', 'learning_rate': 0.1, 'accuracy': 0.5})

	# it is also possible to delete rows
	# sar.delete(('resnet18', 0.1))
