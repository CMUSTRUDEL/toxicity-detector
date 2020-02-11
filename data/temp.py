import glob

all_files = glob.glob("*/*.csv")

for i in all_files:
    f = open(i,encoding='utf-8').read().split("\n")
    f = [i.split(",")[0]+"," for i in f]

    w = open(i,"w")
    w.write("\n".join(f))
    w.close()
