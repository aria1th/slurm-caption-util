import os
files = '~'

for file in os.listdir(files):
    if 'E-vl26B'in file:
        with open(os.path.join(files, file), 'r') as f:
            for line in f:
                if 'PREEMPTION' in line:
                    print(f"Found in {file}: {line}")

# maybe in future, add to file and automate the process?
