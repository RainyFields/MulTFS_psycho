import os
import json

root = os.getcwd()
task = "1back_identity"
for filename in os.listdir(root + "/" + task):
    if filename == "train":
        continue
    f = os.path.join(root + "/" + task, filename)
    if os.path.isdir(f):
        with open(f + "/compo_task_example" ) as f_data:
            data = json.load(f_data)
        data["instruction"] = "is the object the same as the one before? If there is a match, press T, otherwise, press F"
        data.pop("instructions", None)
        with open(f + "/compo_task_example", "w" ) as f_data:
            json.dump(data, f_data)