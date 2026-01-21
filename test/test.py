import sys
import importlib.metadata
print(sys.version)




wanted_libs = {"pandas","numpy","scipy","scikit-learn","matplotlib","seaborn","statsmodels"}
lib_count = 0

#used ai to figure out how to get all installed libraries and write this method call and for loop.
dists = sorted(importlib.metadata.distributions(), key=lambda x: x.metadata['Name'].lower())
for i, dist in enumerate(dists):
        
    name = dist.metadata['Name']
    if name in wanted_libs:
        lib_count += 1
        version = dist.version
        print(f"- {name}: {version}")

if lib_count == len(wanted_libs):
    print("all libs found")
else:
    print("missing libs")
