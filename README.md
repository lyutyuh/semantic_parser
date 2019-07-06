# semantic_parser

Code for final project. 

## Get Google KG data
```python
api_key = "<your_api_key>" # ?
query = 'running man'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    'limit': 10,
    'indent': True,
    'key': api_key,
}

url = service_url + '?' + urllib.parse.urlencode(params)

while True:
    try:
        response = json.loads(urllib.request.urlopen(url).read())
        break
    except Exception as e:
        if "429" in str(e): # too many requests... wait
            time.sleep(10)
```

## Train the model
allennlp train -s <dump_directory> allennlp/MY_CONFIG/sem_parser.config     # single-relation model training
allennlp train -s <dump_directory> allennlp/MY_CONFIG/sem_parser_cvt.config # cvt model training
