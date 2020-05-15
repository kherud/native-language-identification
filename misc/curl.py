import requests

# > docker run -p 8081:8081 allenai/spv2:2.10
# curl -v --data-binary @paper.pdf "http://localhost:8081/v1/json/pdf"
url = "http://localhost:8081/v1/json/pdf"
payload = open("../data/pdfs/AAAI12-4.pdf", "rb")
res = requests.post(url, data=payload)

print(res.json())
