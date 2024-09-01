import streamlit as st
import json
import time
import requests

endpoint = "https://3ttv5myencxjecbxpf7uzuwtzi0tpihp.lambda-url.us-east-1.on.aws/"

def get_response(text):
    payload = {
        "prompt": f"Please summarize the following text.\n {text}"
    }

    response = requests.post(endpoint, json=payload)
    print(f"Reponse Status: {response.status_code}")
    response_text = response.text

    for word in response_text.split():
        yield word + " "
        time.sleep(0.1)


def main():
    st.set_page_config(" Text Summarization Bedrock Cohere")
    st.header("AWS Bedrock integration with Serverless Lambda for Text Summarization with Cohere")

    text = st.text_area("Write text to summarize")

    if st.button("Summarize It!"):
        with st.spinner("processing request.... "):
            st.write(get_response(text))



if __name__ == "__main__":
    main()