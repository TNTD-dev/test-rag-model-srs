import requests
import json

url = "http://127.0.0.1:8001/generate-srs"
payload = {
    "project_description": "A simple calculator app with basic arithmetic operations."
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Response received successfully!")
        srs_content = data['srs_content']
        rag_context = data.get('rag_context') # Capture Context
        
        print("SRS Content Preview:")
        print(srs_content[:500] + "...")
        print(f"RAG Context Available: {bool(rag_context)}")

        # Now test evaluation
        print("\n--- Testing Evaluation Endpoint ---")
        eval_url = "http://127.0.0.1:8001/evaluate-srs"
        eval_payload = {
            "srs_content": srs_content,
            "rag_context": rag_context # Pass context for Faithfulness check
        }
        print(f"Sending request to {eval_url}...")
        eval_response = requests.post(eval_url, json=eval_payload, headers=headers)
        print(f"Eval Status Code: {eval_response.status_code}")
        
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            print("Evaluation received successfully!")
            print("Evaluation Result (Structured):")
            print(json.dumps(eval_data['evaluation_result'], indent=2, ensure_ascii=False))
        else:
             print(f"Error evaluating: {eval_response.text}")

    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception calling API: {e}")
