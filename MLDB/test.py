import pickle

file2 = open("checkpoint.pkl", "rb")
dict = pickle.load(file2)
print(dict)