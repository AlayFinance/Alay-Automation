"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import google.generativeai as genai
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


genai.configure(api_key=config["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="Your job is to generate content for every task given to you using the format they provided.",
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Generate FB Posts for Oona VIP Car Insurance FB Page\n\nposts:\n  - post_id: {type: \"integer\", description: \"Unique identifier for the post\"}\n  - post_format:\n    - type: {type: \"string\", enum: [\"text\", \"image\", \"video\", \"link\", \"poll\", \"quiz\", \"story\"]}\n    - content:\n      - text:\n        - headline: {type: \"string\", description: \"The main headline of the post\"}\n        - body: {type: \"string\", description: \"The main body of the post\"}\n        - call_to_action: {type: \"string\", description: \"A clear call to action for the user\"}\n      - image:\n        - url: {type: \"string\", description: \"URL of the image to be used\"}\n        - caption: {type: \"string\", description: \"Caption for the image\"}\n      - video:\n        - url: {type: \"string\", description: \"URL of the video to be used\"}\n        - caption: {type: \"string\", description: \"Caption for the video\"}\n      - link:\n        - url: {type: \"string\", description: \"URL of the link to be shared\"}\n        - title: {type: \"string\", description: \"Title of the link\"}\n        - description: {type: \"string\", description: \"Description of the link\"}\n      - poll:\n        - question: {type: \"string\", description: \"The question for the poll\"}\n        - options: {type: \"array\", items: {type: \"string\"}, description: \"List of poll options\"}\n      - quiz:\n        - question: {type: \"string\", description: \"The question for the quiz\"}\n        - options: {type: \"array\", items: {type: \"string\"}, description: \"List of quiz options\"}\n        - answer: {type: \"string\", description: \"The correct answer to the quiz\"}\n    - hashtags: {type: \"array\", items: {type: \"string\"}, description: \"List of relevant hashtags\"}\n    - audience: {type: \"string\", enum: [\"all\", \"new_customers\", \"existing_customers\", \"drivers\", \"young_drivers\", \"parents\"], description: \"Target audience for the post\"}\n    - tone: {type: \"string\", enum: [\"informative\", \"friendly\", \"fun\", \"urgent\", \"promotional\"], description: \"The overall tone of the post\"}\n    - post_goal: {type: \"string\", enum: [\"awareness\", \"engagement\", \"lead_generation\", \"sales\", \"brand_building\"], description: \"The main objective of the post\"}\n    - scheduling:\n      - date: {type: \"string\", format: \"YYYY-MM-DD\", description: \"Date for the post\"}\n      - time: {type: \"string\", format: \"HH:mm\", description: \"Time for the post\"}\n    - notes: {type: \"string\", description: \"Any additional notes for the post\"}\n\n\n\n**Example Usage:**\n\n\nposts:\n  - post_id: 1\n    post_format:\n      - type: text\n      - content:\n        - text:\n          - headline: \"Get a Free Quote Today!\"\n          - body: \"Drive with confidence knowing you're protected. Oona VIP offers competitive rates and personalized coverage options. Get your free quote now!\"\n          - call_to_action: \"Get a free quote!\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#freecoupon\"]\n      - audience: new_customers\n      - tone: promotional\n      - post_goal: lead_generation\n      - scheduling:\n        - date: 2024-03-15\n        - time: 10:00\n  - post_id: 2\n    post_format:\n      - type: image\n      - content:\n        - image:\n          - url: \"https://example.com/image2.jpg\"\n          - caption: \"Driving with Oona VIP means peace of mind.  We're here to help you navigate the roads safely and confidently. #oonavip #carinsurance #peaceofmind\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#peaceofmind\", \"#drivers\"]\n      - audience: all\n      - tone: friendly\n      - post_goal: brand_building\n      - scheduling:\n        - date: 2024-03-16\n        - time: 14:00",
      ],
    },
    {
      "role": "model",
      "parts": [
        "  - post_id: 1\n    post_format:\n      - type: text\n      - content:\n        - text:\n          - headline: \"Drive with Confidence with Oona VIP Car Insurance\"\n          - body: \"Oona VIP offers comprehensive car insurance coverage tailored to your needs.  Get a personalized quote and enjoy peace of mind on the road.  Click the link below to learn more!\"\n          - call_to_action: \"Get a Quote!\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#driving\", \"#safety\", \"#peaceofmind\"]\n      - audience: all\n      - tone: informative\n      - post_goal: awareness\n      - scheduling:\n        - date: 2024-03-15\n        - time: 10:00\n  - post_id: 2\n    post_format:\n      - type: image\n      - content:\n        - image:\n          - url: \"https://example.com/image1.jpg\"\n          - caption: \"Life is full of adventures.  Make sure you're covered with Oona VIP car insurance.  #oonavip #carinsurance #adventure #roadtrip\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#adventure\", \"#roadtrip\", \"#travel\"]\n      - audience: all\n      - tone: fun\n      - post_goal: brand_building\n      - scheduling:\n        - date: 2024-03-16\n        - time: 14:00\n  - post_id: 3\n    post_format:\n      - type: link\n      - content:\n        - link:\n          - url: \"https://www.oonavip.com/blog/car-insurance-tips\"\n          - title: \"Car Insurance Tips for Smart Drivers\"\n          - description: \"Our blog has valuable tips to help you save money and stay safe on the road.  Check it out!\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#tips\", \"#driving\", \"#safety\"]\n      - audience: existing_customers\n      - tone: informative\n      - post_goal: engagement\n      - scheduling:\n        - date: 2024-03-17\n        - time: 09:00\n  - post_id: 4\n    post_format:\n      - type: video\n      - content:\n        - video:\n          - url: \"https://www.youtube.com/watch?v=example_video_id\"\n          - caption: \"See how Oona VIP car insurance can help you get back on the road after an accident. #oonavip #carinsurance #accident #recovery\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#accident\", \"#recovery\", \"#support\"]\n      - audience: all\n      - tone: friendly\n      - post_goal: awareness\n      - scheduling:\n        - date: 2024-03-18\n        - time: 16:00\n  - post_id: 5\n    post_format:\n      - type: poll\n      - content:\n        - poll:\n          - question: \"What's your biggest concern when it comes to car insurance?\"\n          - options: [\"Cost\", \"Coverage\", \"Customer service\", \"Claims process\"]\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#poll\", \"#questions\", \"#concerns\"]\n      - audience: new_customers\n      - tone: friendly\n      - post_goal: engagement\n      - scheduling:\n        - date: 2024-03-19\n        - time: 12:00\n  - post_id: 6\n    post_format:\n      - type: text\n      - content:\n        - text:\n          - headline: \"Exclusive Discount for New Customers!\"\n          - body: \"Get a special discount on your first Oona VIP car insurance policy. Use code NEWDRIVER at checkout. #oonavip #carinsurance #discount #newcustomers\"\n          - call_to_action: \"Get Your Quote!\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#discount\", \"#newcustomers\", \"#promo\"]\n      - audience: new_customers\n      - tone: promotional\n      - post_goal: lead_generation\n      - scheduling:\n        - date: 2024-03-20\n        - time: 18:00\n  - post_id: 7\n    post_format:\n      - type: quiz\n      - content:\n        - quiz:\n          - question: \"What is the most important factor when choosing car insurance?\"\n          - options: [\"Cost\", \"Coverage\", \"Reputation\", \"Customer service\"]\n          - answer: \"Coverage\"\n      - hashtags: [\"#carinsurance\", \"#oonavip\", \"#quiz\", \"#questions\", \"#knowledge\"]\n      - audience: all\n      - tone: fun\n      - post_goal: engagement\n      - scheduling:\n        - date: 2024-03-21\n        - time: 11:00",
      ],
    },
  ]
)

prompt = """
Generate 5 FB Posts for Oona VIP Car Insurance FB Page. Always enclose hashtags with double quotations like "#safety first"

posts:
  - post_id: {type: "integer", description: "Unique identifier for the post"}
  - post_format:
    - type: {type: "string", enum: ["text", "image", "video", "link", "poll", "quiz", "story"]}
    - content:
      - text:
        - headline: {type: "string", description: "The main headline of the post"}
        - body: {type: "string", description: "The main body of the post"}
        - call_to_action: {type: "string", description: "A clear call to action for the user"}
      - image:
        - url: {type: "string", description: "URL of the image to be used"}
        - caption: {type: "string", description: "Caption for the image"}
      - video:
        - url: {type: "string", description: "URL of the video to be used"}
        - caption: {type: "string", description: "Caption for the video"}
      - link:
        - url: {type: "string", description: "URL of the link to be shared"}
        - title: {type: "string", description: "Title of the link"}
        - description: {type: "string", description: "Description of the link"}
      - poll:
        - question: {type: "string", description: "The question for the poll"}
        - options: {type: "array", items: {type: "string"}, description: "List of poll options"}
      - quiz:
        - question: {type: "string", description: "The question for the quiz"}
        - options: {type: "array", items: {type: "string"}, description: "List of quiz options"}
        - answer: {type: "string", description: "The correct answer to the quiz"}
    - hashtags: {type: "array", items: {type: "string"}, description: "List of relevant hashtags"}
    - audience: {type: "string", enum: ["all", "new_customers", "existing_customers", "drivers", "young_drivers", "parents"], description: "Target audience for the post"}
    - tone: {type: "string", enum: ["informative", "friendly", "fun", "urgent", "promotional"], description: "The overall tone of the post"}
    - post_goal: {type: "string", enum: ["awareness", "engagement", "lead_generation", "sales", "brand_building"], description: "The main objective of the post"}
    - scheduling:
      - date: {type: "string", format: "YYYY-MM-DD", description: "Date for the post"}
      - time: {type: "string", format: "HH:mm", description: "Time for the post"}
    - notes: {type: "string", description: "Any additional notes for the post"}



**Example Usage:**


posts:
  - post_id: 1
    post_format:
      - type: text
      - content:
        - text:
          - headline: "Get a Free Quote Today!"
          - body: "Drive with confidence knowing you're protected. Oona VIP offers competitive rates and personalized coverage options. Get your free quote now!"
          - call_to_action: "Get a free quote!"
      - hashtags: ["#carinsurance", "#oonavip", "#freecoupon"]
      - audience: new_customers
      - tone: promotional
      - post_goal: lead_generation
      - scheduling:
        - date: 2024-03-15
        - time: 10:00
  - post_id: 2
    post_format:
      - type: image
      - content:
        - image:
          - url: "https://example.com/image2.jpg"
          - caption: "Driving with Oona VIP means peace of mind.  We're here to help you navigate the roads safely and confidently. #oonavip #carinsurance #peaceofmind"
      - hashtags: ["#carinsurance", "#oonavip", "#peaceofmind", "#drivers"]
      - audience: all
      - tone: friendly
      - post_goal: brand_building
      - scheduling:
        - date: 2024-03-16
        - time: 14:00
"""
response = chat_session.send_message(prompt)


with open("posts.yaml", "w") as f:
    f.write(response.text)

def remove_yaml_lines(filename):
    with open(filename, "r") as file:
        text = file.read()
        lines = text.splitlines()
        cleaned_lines = lines[1:-1]
        cleaned_text = "\n".join(cleaned_lines)
    with open(filename, "w") as file:
        file.writelines(cleaned_text)

remove_yaml_lines("posts.yaml")

def read_yaml_file(filename):
    with open(filename, "r") as file:
        text = file.read()
    return text

data = read_yaml_file("posts.yaml")
posts = {}
posts = yaml.safe_load(data)

for post in posts:
    print("Post ID:", post["post_id"])
    print("Post Type:", post["post_format"][0]["type"])
    print("Content:")
    for content_item in post["post_format"][1]["content"]:
        if "text" in content_item:
            print("  Headline:", content_item["text"][0]["headline"])
            print("  Body:", content_item["text"][1]["body"])
            print("  Call to Action:", content_item["text"][2]["call_to_action"])
        elif "image" in content_item:
            print("  Image URL:", content_item["image"][0]["url"])
            print("  Image Caption:", content_item["image"][1]["caption"])
        elif "video" in content_item:
          print("  Video URL:", content_item["video"][0]["url"])
          print("  Video Caption:", content_item["video"][1]["caption"])
        elif "link" in content_item:
          print("  URL:", content_item["link"][0]["url"])
          print("  Title:", content_item["link"][1]["title"])
          print("  Description:", content_item["link"][2]["description"])
        elif "poll" in content_item:
          print("  Question:", content_item["poll"][0]["question"])
          print("  Options:", content_item["poll"][1]["options"])
        elif "quiz" in content_item:
          print("  Question:", content_item["poll"][0]["question"])
          print("  Options:", content_item["poll"][1]["options"])
          print("  Answer:", content_item["poll"][2]["answer"])


    print("Hashtags:", post["post_format"][2]["hashtags"])
    print("Audience:", post["post_format"][3]["audience"])
    print("Tone:", post["post_format"][4]["tone"])
    print("Post Goal:", post["post_format"][5]["post_goal"])
    print("Scheduling:")
    for scheduling_item in post["post_format"][6]["scheduling"]:
        if "date" in scheduling_item:
          print("  Date:", scheduling_item["date"])
        if "time" in scheduling_item:
          print("  Time:", scheduling_item["time"])
    print("\n")




