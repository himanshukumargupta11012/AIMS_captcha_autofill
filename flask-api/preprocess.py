import os

for i in os.listdir("./data"):
    if ")." in i:
        os.remove(os.path.join("./data", i))
        print(i)
    