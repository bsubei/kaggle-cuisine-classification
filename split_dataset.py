import pandas as pd

data_folder = 'data/'
all_train = pd.read_json(data_folder + 'train.json')

# write out first half of training set
all_train[0: all_train.shape[0]/2].to_json(data_folder + 'train1.json')
# then write out second half
all_train[all_train.shape[0]/2:].to_json(data_folder + 'train2.json')
