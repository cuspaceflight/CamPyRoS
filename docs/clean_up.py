import os

for file in os.listdir("build/html"):
    # Read in the file
    print(file)
    if file[0] not in [".","_"] and file[-3:] not in ["inv"]:
        route="build/html/%s"%file
        with open(route, "r") as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace("trajectory.main", "trajectory")

        # Write the file out again
        with open(route, "w") as file:
            file.write(filedata)