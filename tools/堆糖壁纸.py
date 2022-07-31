from ntpath import join
from matplotlib.pyplot import text
import requests
import re
import json
import pprint
url = "https://www.duitang.com/album/?id=86839967#!albumpics"
header = {"user-agent":"Mozilla/5.0"}
r = requests.get(url=url,headers=header)

message = re.findall('src="(.*?).jpeg"',r.text)
message = set([f"{i}.jpeg" for i in message])
# for i,img in enumerate(message):
#     with open(f"test/{i}.jpeg","wb") as f:
#         m = requests.get(url=img,headers=header)
#         f.write(m.content)
#         print(i)
print(list(message))

# message = re.findall('<script>window.__playinfo__=(.*?)</script>',r.text)[0]
# json_message = json.loads(message)
# pprint.pprint(json_message)

# r2 = requests.get("https://cn-cq-gd-bcache-18.bilivideo.com/upgcxcode/25/81/391298125/391298125_x2-1-30033.m4s?e=ig8euxZM2rNcNbdlhoNvNC8BqJIzNbfqXBvEqxTEto8BTrNvN0GvT90W5JZMkX_YN0MvXg8gNEV4NC8xNEV4N03eN0B5tZlqNxTEto8BTrNvNeZVuJ10Kj_g2UB02J0mN0B5tZlqNCNEto8BTrNvNC7MTX502C8f2jmMQJ6mqF2fka1mqx6gqj0eN0B599M=&uipk=5&nbs=1&deadline=1658945870&gen=playurlv2&os=bcache&oi=1966845351&trid=00006eb692189b2245a3b2cd1ea888c94020u&mid=0&platform=pc&upsig=0af0fa9a5e479139751e05cf6b41b1c4&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform&cdnid=20313&bvc=vod&nettype=0&orderid=0,3&agrr=1&bw=36333&logo=80000000'",headers=header)
# print(r2.text)