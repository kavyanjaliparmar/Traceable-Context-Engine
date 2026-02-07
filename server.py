import os
import json
import io
import fitz  # PyMuPDF
import google.generativeai as genai
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")

def get_model(model_name="gemini-1.5-flash"):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please create a .env file and set GOOGLE_API_KEY.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def extract_and_tag_pdf(file_stream):
    """
    Extracts text from a PDF and tags each paragraph with [[P{page}_{block}]].
    Returns a string of tagged text and a dictionary mapping tags to raw text.
    """
    doc = fitz.open(stream=file_stream, filetype="pdf")
    tagged_text = ""
    source_map = {}
    page_count = len(doc)

    for page_num, page in enumerate(doc):
        p_num = page_num + 1
        blocks = page.get_text("blocks")
        for i, block in enumerate(blocks):
            if block[6] == 0:  # Text block
                text = block[4].strip()
                if text:
                    tag = f"[[P{p_num}_{i}]]"
                    source_map[tag] = text
                    tagged_text += f"{tag} {text}\n\n"

    return tagged_text, source_map, page_count

def summarize_with_gemini(text, model_name="gemini-1.5-flash"):
    """
    Summarizes text using Gemini with structured JSON output.
    """
    model = get_model(model_name)

    generation_config = {
        "temperature": 0.5,
        "response_mime_type": "application/json"
    }

    prompt = (
        "You are an expert analyst. Your task is to compress the provided document into a structured summary.\n"
        "Input text contains source tags like [[P1_0]], [[P2_5]] at the start of blocks.\n"
        "You MUST preserve these tags in your output for traceability.\n\n"
        "Output Format (JSON):\n"
        "{\n"
        "  \"summary\": {\n"
        "    \"high_level_summary\": \"A concise 2-3 sentence overview of the entire document.\",\n"
        "    \"sections\": [\n"
        "      {\n"
        "        \"title\": \"Section/Chapter Title\",\n"
        "        \"key_points\": [\n"
        "          {\n"
        "            \"statement\": \"Key fact or claim.\",\n"
        "            \"source_ids\": [\"[[P1_0]]\", \"[[P1_2]]\"],\n"
        "            \"risk_type\": \"None/Operational/Financial/Legal\",\n"
        "            \"details\": \"A comprehensive 3-5 sentence deep-dive. Include background context, specific figures, dates, exceptions, and potential implications.\",\n"
        "            \"rationale\": \"Why this retention is critical.\"\n"
        "          }\n"
        "        ]\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"meta_analysis\": {\n"
        "    \"omitted_themes\": [\n"
        "      {\n"
        "        \"theme\": \"Description of omitted topic\",\n"
        "        \"reason_for_omission\": \"Why it was removed.\",\n"
        "        \"impact_score\": \"Low/Medium/High\"\n"
        "      }\n"
        "    ],\n"
        "    \"global_retention_rationale\": \"Overall strategy.\"\n"
        "  }\n"
        "}\n\n"
        f"Document Content:\n{text}"
    )

    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/process', methods=['POST'])
def process_document():
    print("\n--- NEW PROCESS REQUEST ---")
    if 'file' not in request.files:
        print("Error: No file in request")
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({"success": False, "error": "No file selected"}), 400

    try:
        print(f"Processing file: {file.filename}")
        file_stream = file.read()
        tagged_text, source_map, page_count = extract_and_tag_pdf(file_stream)
        print(f"Extracted {len(tagged_text)} characters from {page_count} pages.")
        
        # Select model (can be passed from frontend if needed)
        model_name = "gemini-1.5-flash"
        
        print(f"Calling Gemini ({model_name})...")
        response_json_str = summarize_with_gemini(tagged_text, model_name)
        print("Gemini response received.")
        
        # Strip code blocks if present
        if response_json_str.strip().startswith("```"):
            print("Cleaning up markdown code blocks from response...")
            response_json_str = re.sub(r'^```(json)?\s*|\s*```$', '', response_json_str.strip(), flags=re.MULTILINE)

        data = json.loads(response_json_str)
        print("Successfully parsed JSON data.")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "page_count": page_count,
            "source_map": source_map,
            "tagged_text": tagged_text,
            "data": data
        })

    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    request_data = request.json
    question = request_data.get('question')
    tagged_text = request_data.get('tagged_text') # In a real app, this might be stored in a session or DB
    
    if not question or not tagged_text:
        return jsonify({"success": False, "error": "Missing question or context"}), 400

    try:
        model = get_model("gemini-1.5-flash")
        prompt = (
            "You are an expert document analyst. Answer the user's question based ONLY on the provided text.\n"
            "Use source tags like [[P1_0]] as proofs.\n"
            f"Context:\n{tagged_text}\n\n"
            f"Question: {question}\n\n"
            "Answer (with proofs):"
        )
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        
        # Extract evidence tags (simplified for now)
        evidence = ", ".join(re.findall(r'\[\[P\d+_\d+\]\]', response.text))
        
        return jsonify({
            "success": True,
            "answer": response.text,
            "evidence": evidence or "Based on document context"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9006)
