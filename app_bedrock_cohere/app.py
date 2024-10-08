import streamlit as st
import json
import time
import requests

endpoint = "https://3ttv5myencxjecbxpf7uzuwtzi0tpihp.lambda-url.us-east-1.on.aws/"

def get_response(text):
    payload = {
        "prompt": f"Summarize the following text. Return only the summary, with no additional text or explanations: \n {text}"
    }

    response = requests.post(endpoint, json=payload)
    print(f"Reponse Status: {response.status_code}")
    response_text = response.text

    for word in response_text.split():
        yield word + " "
        time.sleep(0.1)


def main():
    st.set_page_config("Text Summarization with Cohere")
    st.header("Text Summarization with Cohere in AWS Bedrock + Lambda")

    text = st.text_area("Write text to summarize")

    if st.button("Summarize It!"):
        with st.spinner("processing request.... "):
            st.write(get_response(text))



if __name__ == "__main__":
    main()