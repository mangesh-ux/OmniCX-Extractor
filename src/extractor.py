import os
import json
from openai import OpenAI
from schema import LogisticsCXMetrics # Imports your massive schema
import dotenv

dotenv.load_dotenv()  # Loads .env into os.environ (returns None, so don't index it)

# Initialize the OpenAI client (Make sure OPENAI_API_KEY is in your .env or system vars)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_logistics_data(transcript: str) -> LogisticsCXMetrics:
    """Passes the transcript to gpt-4o-mini and forces the output to match our Pydantic schema."""
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."},
            {"role": "user", "content": f"Transcript:\n{transcript}"}
        ],
        response_format=LogisticsCXMetrics,
        temperature=0.1 # Keep it low for highly deterministic, analytical outputs
    )
    
    return completion.choices[0].message.parsed

if __name__ == "__main__":
    # A dummy transcript of a highly frustrated FedEx customer
    test_transcript = """
    Agent: Thank you for calling FedEx, my name is Sarah. How can I help you?
    Customer: Hi Sarah, I'm incredibly frustrated. I've been waiting for my medical supplies all day. The tracking says DEX08, recipient not in, but I have been sitting on my porch for six hours! No one knocked.
    Agent: I am so sorry to hear that, let me look up that 1Z tracking number for you. 
    Customer: It's not a 1Z, it's 9400123456789. 
    Agent: Ah, my apologies. Okay, I see it. It looks like the driver marked it as business closed. I will route a ticket to the local hub to see if they can re-attempt delivery today.
    Customer: Thank you, I really need those supplies. I appreciate you looking into it.
    """
    
    print("Sending transcript to gpt-4o-mini...")
    
    # Run the extraction
    result = extract_logistics_data(test_transcript)
    
    # Print the beautiful, structured JSON
    print("\n--- EXTRACTION RESULT ---")
    print(result.model_dump_json(indent=2))