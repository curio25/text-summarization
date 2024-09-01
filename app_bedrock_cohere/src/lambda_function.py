import json
import boto3

# {
#  "modelId": "cohere.command-text-v14",
#  "contentType": "application/json",
#  "accept": "application/json",
#  "body": "{\"prompt\":\"this is where you place your input text\",\"max_tokens\":200,\"temperature\":0.9,\"p\":1,\"k\":0}"
# }

model_id = "cohere.command-text-v14"
contentType = "application/json"
accept = "application/json"

bedrock_client = boto3.client("bedrock-runtime")

def lambda_handler(event, context):
    # TODO implement
    
    event_body = json.loads(event['body'])
    prompt = event_body.get("prompt")
    
    body = {
        "prompt":prompt,
        "max_tokens":200,
        "temperature":0.5,
        "p":0.9,
        "k":0
    }
    
    payload = {
        "modelId": model_id,
        "contentType": contentType,
        "accept": accept,
        "body": json.dumps(body)
    }
    
    response = bedrock_client.invoke_model(**payload)
    
    print(response)
    
    response_bytes = response["body"].read()
    response_json = json.loads(response_bytes)
    
    print(response_json)
    
    response_text = response_json["generations"][0]['text']
    
    return {
        'statusCode': 200,
        'body': json.dumps(response_text)
    }
