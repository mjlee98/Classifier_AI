import sys
import math

#calculate probability density function
def pdf(x, mean, std):
	exponential = math.exp(-((x - mean)**2) / (2*(std**2)))
	return exponential / (std * math.sqrt(2*math.pi))

def naive_bayes(training_file, testing_file):

	#find number of attributes
	l = training_file.readline()
	training_file.seek(0)
	n = len(l.split(","))

	#initialise standard deviation and mean of each attribute
	yes_mean = [0]*(n-1)
	no_mean = [0]*(n-1)
	yes_std = [0]*(n-1)
	no_std = [0]*(n-1)
	num_yes = 0
	num_no = 0

	#calculate mean
	for line in training_file:
		attributes = line.rstrip().split(",")

		#for each attribute (not including class)
		if (attributes[n-1] == 'yes'):
			num_yes += 1
			for i in range(0, n-1):
				yes_mean[i] += float(attributes[i])
		else:
			num_no += 1
			for i in range(0, n-1):
				no_mean[i] += float(attributes[i])
	for i in range(0, len(yes_mean)):
		yes_mean[i] = yes_mean[i] / num_yes
		no_mean[i] = no_mean[i] / num_no

	#reset training_file pointer
	training_file.seek(0)

	#calculate standard deviation
	for line in training_file:
		attributes = line.rstrip().split(",")

		#for each attribute (not including class)
		for i in range(0, n-1):
			if (attributes[n-1] == 'yes'):
				yes_std[i] += (float(attributes[i]) - yes_mean[i])**2
			else:
				no_std[i] += (float(attributes[i]) - no_mean[i])**2
	for i in range(0, n-1):
		yes_std[i] = math.sqrt(yes_std[i] / (num_yes-1))
		no_std[i] = math.sqrt(no_std[i] / (num_no-1))

	#use naive bayes on test data
	for line in testing_file:
		attributes = line.rstrip().split(",")

		#probability of yes and no
		probability_yes = 1
		probability_no = 1

		#calculate probability for each line
		for i in range(0, n-1):
			probability_yes *= pdf(float(attributes[i]), yes_mean[i], yes_std[i])
			probability_no *= pdf(float(attributes[i]), no_mean[i], no_std[i])
		probability_yes *= (num_yes / (num_yes + num_no))
		probability_no *= (num_no / (num_yes + num_no))

		if (probability_yes >= probability_no):
			print('yes')
		else:
			print('no')

class Node:
	def __init__(self,attribute):
		self.attribute = attribute
		self.low = []
		self.medium = []
		self.high = []
		self.very_high = []

def entropy(data):
	#count yes/no ratio
	yes = 0
	no = 0
	for row in data:
		if row[-1] == "yes":
			yes += 1
		elif row[-1] == "no":
			no += 1
		else:
			raise Exception("Invalid class")

	P_yes = yes/(yes+no)
	P_no = no/(yes+no)
	if yes == 0 and no != 0:
		ent = - 0 - P_no*math.log(P_no,2)
	elif yes != 0 and no == 0:
		ent = - P_yes*math.log(P_yes,2) - 0
	else:
		ent = - P_yes*math.log(P_yes,2) - P_no*math.log(P_no,2)
	return ent

#choose the attribute with most information gain
def choose_attribute(data,attributes_to_use):

	max_attribute = attributes_to_use[0]
	max_info_gain = -1

	#current entropy
	T1 = entropy(data)

	#calculate entropy of each attribute
	for attribute_no in attributes_to_use:
		#entropy after split
		low = []
		med = []
		high = []
		v_high = []
		for row in data:
			if row[attribute_no] == "low":
				low.append(row)
			elif row[attribute_no] == "medium":
				med.append(row)
			elif row[attribute_no] == "high":
				high.append(row)
			elif row[attribute_no] == "very high":
				v_high.append(row)
			else:
				raise Exception("Invalid attribute in attribute ",attribute_no)
		T2 = 0
		if len(low) != 0:
			T2 += len(low)*entropy(low) / len(data)
		if len(med) != 0:
			T2 += len(med)*entropy(med) / len(data)
		if len(high) != 0:
			T2 += len(high)*entropy(high) / len(data)
		if len(v_high) != 0:
			T2 += len(v_high)*entropy(v_high) / len(data)
		if T1 - T2 > max_info_gain:
			max_attribute = attribute_no
			max_info_gain = T1 - T2
	return max_attribute

def decision_tree_recursion(data,attributes_to_use,default):

		if len(data) == 0:
			return default

		else:

			all_same = True
			yn = data[0][-1]
			for row in data:
				if row[-1] != yn:
					all_same = False

			#if all classes are same
			if all_same:
				return yn

			elif len(attributes_to_use) == 0:
				yes = 0
				no = 0
				for row in data:
					if row[-1] == "yes":
						yes += 1
					elif row[-1] == "no":
						no += 1
					else:
						raise Exception("invalid class")
				if yes >= no:
					return "yes"
				else:
					return "no"



			else:
				best = choose_attribute(data, attributes_to_use)
				split_node = Node(best)
				low = []
				medium = []
				high = []
				very_high = []

				for row in data:
					if row[best] == "low":
						low.append(row)
					elif row[best] == "medium":
						medium.append(row)
					elif row[best] == "high":
						high.append(row)
					elif row[best] == "very high":
						very_high.append(row)
					else:
						raise Exception("Something went wrong")

				new_attributes = []
				for attribute in attributes_to_use:
					if attribute != best:
						new_attributes.append(attribute)


				#recursion
				yes = 0
				no = 0
				for row in data:
					if row[-1] == "yes":
						yes += 1
					elif row[-1] == "no":
						no += 1
					else:
						raise Exception("invalid class")
				if yes >= no:
					default = "yes"

				else:
					default = "no"

				split_node.low = decision_tree_recursion(low,new_attributes,default)
				split_node.medium = decision_tree_recursion(medium,new_attributes,default)
				split_node.high = decision_tree_recursion(high,new_attributes,default)
				split_node.very_high = decision_tree_recursion(very_high,new_attributes,default)

				return split_node

def decision_tree(training_file, testing_file):

	#Data stored in 2D array
	data = training_file.readlines()
	i = 0
	while i < len(data):
		data[i] = data[i].rstrip().split(',')
		i += 1

	samples = testing_file.readlines()
	i = 0
	while i < len(samples):
		samples[i] = samples[i].rstrip().split(',')
		i += 1

	attributes_to_use = []
	j = 0
	while j < len(data[0]) - 1:
		attributes_to_use.append(j)
		j += 1

	yes = 0
	no = 0
	for row in data:
		if row[-1] == "yes":
			yes += 1
		else:
			no +=1
	if yes >= no:
		dt = decision_tree_recursion(data,attributes_to_use,"yes")
	else:
		dt = decision_tree_recursion(data,attributes_to_use,"no")


	dt = dt.very_high.high.high.high.low.low.high

	print("root",dt.attribute)
	if type(dt.low) == str:
		print(dt.low)
	else:
		print(dt.low.attribute)

	if type(dt.medium) == str:
		print(dt.medium)
	else:
		print(dt.medium.attribute)

	if type(dt.high) == str:
		print(dt.high)
	else:
		print(dt.high.attribute)

	if type(dt.very_high) == str:
		print(dt.very_high)
	else:
		print(dt.very_high.attribute)





#command line arguments
training = sys.argv[1]
testing = sys.argv[2]
algorithm = sys.argv[3]

#open files
training_file = open(training,"r")
testing_file = open(testing,"r")

if (algorithm == 'NB'):
	naive_bayes(training_file, testing_file)
else:
	decision_tree(training_file, testing_file)

#close files
training_file.close()
testing_file.close()
