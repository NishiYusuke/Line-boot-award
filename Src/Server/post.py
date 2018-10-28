import json
import urllib.request


def post(user_id):
	url = "http://localhost:3000/users/"+str(user_id)+"/in_or_out"
	print(url)
	data = {
    	"in_or_out": True,
	}
	print(json.dumps(data).encode())
	headers = {
    	"Content-Type" : "application/json"
	}
	req = urllib.request.Request(url, json.dumps(data).encode("utf-8"), headers)
	with urllib.request.urlopen(req) as res:
	    body = res.read()
post(1)

