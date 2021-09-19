import twitter
import csv
api_key=""
api_secret_key=""
bearer_token=""
access_token=""
access_token_secrets=""
api = twitter.Api(consumer_key=api_key,consumer_secret=api_secret_key,access_token_key=access_token,access_token_secret=access_token_secrets)
# print(api.VerifyCredentials())
# statuses = api.GetUserTimeline(1266645527917211649)
# print([s.text for s in statuses])
# tweets=api.GetSearch(term="covid", since=2020-11-18,count=10)
lat=input(" Latitiude ")
lon=input(" Longitude ")
rad=input(" Radius ")
query=lat+','+lon+','+rad+"mi"
cou=int(input("Number of tweets"))
ter=input(" Related term ")
dat=input(" Date should be formatted as YYYY-MM-DD. ")
tweets=api.GetSearch(geocode=query,lang='en',count=cou,term=ter,since=dat)
# tweets.text.encode("utf-8")
a=[t.text for t in tweets]
# a.encode("utf-8")
# print(a)
fields = ['Name']
with open('tweets.csv','a',newline='',encoding='utf-8') as f:
    write = csv.writer(f) 
    # write.writerow(fields) 
    for sen in a:
        write.writerow([sen]) 
