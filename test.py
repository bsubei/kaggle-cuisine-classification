import json

# opens the JSON file and reads it, and stores it in my_data
my_data = json.loads(open("train.json").read())

# prints length of JSON entries
print len(my_data)

# prints first JSON entry
print json.dumps(my_data[0])

# prints last JSON entry
print json.dumps(my_data[-1])
