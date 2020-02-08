import get_data

ids = ['zfs/zfsonlinux/8213', 'openusercss.org/OpenUserCSS/121', 'bootstrap/twbs/19636', 'bootstrap/twbs/13476', 'bootstrap/twbs/19651', 'bootstrap/twbs/1009', 'mongo-express/mongo-express/411', 'angular-calendar/mattlewis92/493', 'dialogflow-nodejs-client-v2/dialogflow/63']

print("issue_number,toxic,version,url")

for id in ids:
    formatted = id.split("/")
    print("{},{},{},{}".format(formatted[-1],'y','v10',"https://api.github.com/repos/"+formatted[1]+"/"+formatted[0]))

