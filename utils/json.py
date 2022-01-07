# Contains json support functions.

import json

def save_json(name, array):
    file = open(name + '.json', "w")
    json.dump(array, file, indent=4)
    file.close()

def open_json(name):
    return json.loads(open(name + ".json", "r").read())

