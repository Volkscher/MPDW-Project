import requests
import time
import os
import re
from collections import Counter
from ollama import Client
from PIL import Image

API_KEY = "GQLjmcqR-6b_d3P38AOEUQ"  # Replace with your Horde key
#PROMPT = "A breathtaking scenic landscape featuring rolling green hills, a clear blue sky with soft clouds, a calm river flowing through the valley, and distant snow-capped mountains, ultra-realistic lighting, 8k resolution, golden hour glow, hyper-detailed, photorealistic style"

def generate_image():
  # Step 1: Submit generation request
  payload = {
      "prompt": PROMPT,
      "params": {
          "n": 1,
          "width": 512,
          "height": 512,
          "models": ["stable_diffusion"]
      }
  }
  headers = {
      "apikey": API_KEY,
      "Client-Agent": "YourApp:0.1"
  }
  resp = requests.post("https://stablehorde.net/api/v2/generate/async", json=payload, headers=headers)
  job_id = resp.json()["id"]

  # Step 2: Poll until complete
  while True:
      r = requests.get(f"https://stablehorde.net/api/v2/generate/status/{job_id}")
      data = r.json()
      if data.get("done"):
          break
      time.sleep(5)

  # Step 3: Get image URL(s)
  image_urls = data["generations"][0]["img"]
  print("Image:", image_urls)

  # Step 4: Download image to current directory
  image_data = requests.get(image_urls).content
  filename = "generated_image.png"
  with open(filename, "wb") as f:
      f.write(image_data)

  print(f"Image saved as {os.path.abspath(filename)}")


  # Load the PNG image
  png_image = Image.open("generated_image.png").convert("RGB")  # Convert to RGB to remove alpha

  # Save as JPG
  png_image.save("converted_image.jpg", "JPEG")


# Generate 5 questions from LLaVA
def generate_questions():
  response = client.chat(
      model=model_multimodal,
      messages=[{
          'role': 'user',
          'content': (
              'Generate 5 simple yes or no questions that test basic visual facts about this image. '
              'Focus on clear, elementary observations such as what objects, people, scenery, or colors are visible. '
              'Avoid detailed actions or assumptions about what characters are doing. '
              'Format the questions as a numbered list (e.g., "1. ..."), and ensure each one can be answered with only "yes" or "no".'
          ),
          'images': [img_path]
      }]
  )
  text = response.message.content
  parts = re.split(r'\d+\.\s+', text)
  return [q.strip() for q in parts if q.strip()]


def generate_question():
  client = Client(
      host='https://twiz.novasearch.org/ollama',
      headers={'x-some-header': 'some-value'}
  )

  model_multimodal = 'llava-phi3:latest'
  img_path = './converted_image.jpg'
  image = Image.open(img_path)

  # Extract yes/no
  def extract_yes_no(text):
      match = re.search(r'\b(yes|no)\b', text.lower())
      return match.group(1) if match else "unknown"

  # Ask a question multiple times and get majority
  def get_majority_answer(question, img_path, trials=5):
      responses = []
      for _ in range(trials):
          response = client.chat(
              model=model_multimodal,
              messages=[{
                  'role': 'user',
                  'content': f"Answer with yes or no: {question}",
                  'images': [img_path]
              }]
          )
          answer_text = response.message.content
          responses.append(extract_yes_no(answer_text))
      counts = Counter(responses)
      majority, count = counts.most_common(1)[0]
      return majority if majority in ["yes", "no"] else "unknown", responses


  # Full verification loop
  verified_questions = []
  max_attempts = 10  # prevent infinite loops

  while len(verified_questions) < 5 and max_attempts > 0:
      needed = 5 - len(verified_questions)
      print(f"\n⏳ Generating {needed} new question(s)...")
      candidate_questions = generate_questions()

      for q in candidate_questions:
          if len(verified_questions) == 5:
              break
          majority, responses = get_majority_answer(q, img_path)
          print(f"\nQ: {q}\n→ Answers: {responses}\n→ Inferred: {majority}")
          if majority in ["yes", "no"]:
              verified_questions.append({
                  "question": q,
                  "inferred_answer": majority,
                  "responses": responses
              })

      max_attempts -= 1

  if len(verified_questions) < 5:
      print("\n❌ Failed to generate 5 valid questions after multiple attempts.")
  else:
      print("\n✅ Final set of 5 answerable yes/no questions:")
      for q in verified_questions:
          print(f"- {q['question']} (→ {q['inferred_answer']})")  

def execute_clip_query():
    print('Not Yet')


def setup_phase3():
  # Ask to the user to insert a PROMT to generate an image
  print("Please enter a prompt to generate an image:")
  prompt = input("Prompt: ")
  global PROMPT
  PROMPT = prompt

  generate_image()
  generate_question()
  execute_clip_query()



