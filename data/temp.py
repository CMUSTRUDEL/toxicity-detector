import glob

all_files = glob.glob("languages/*.csv")

for i in ['languages\\python_issues.csv', 'languages\\ruby_issues.csv', 'languages\\r_issues.csv']:
    text_file_location = i.split("_")[0]+"_results.txt"
    f2 = open(text_file_location).read().split("\n")[1:]

    d = {}
    for j in f2:
        if "/" in j:
            repo_name = j.split(" ")[0]
            repo_name = repo_name.split("/")
            repo_name = repo_name[1]+"/"+repo_name[0]+"/"+repo_name[2]

            toxic = ['n','y'][int(j.split(" ")[1])]
            d[repo_name] = toxic

    f = open(i).read().split("\n")
    f[0]+=",toxicity"
    for j in range(len(f)):
       if "/" in f[j]:
           f[j]+=","+d[f[j].split(",")[0]]

    w = open(i,"w",encoding='utf-8')
    w.write("\n".join(f))
    w.close()
