import ctypes
import os
from bs4 import BeautifulSoup
import pandas as pd
import pyautogui
import qrcode
import numpy as np
import torch
import pywhatkit
import subprocess
import threading
import random
import pygame
import winshell
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import noisereduce as nr
import pyjokes
import shutil
import time
import edge_tts
import asyncio
from clint.textui import progress
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests

data = pd.read_csv(r'E:\New folder\commands1.csv')
commands = data['Command'].values
intents = data['Intent'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)

X_train, X_test, y_train, y_test = train_test_split(commands, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def listen():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 4000  # Adjust this threshold based on your environment

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)

        while True:
            print("Listening....", end='', flush=True)
            try:
                audio = recognizer.listen(source, timeout=None)
                print("\rRecognizing...   ", end='', flush=True)
                
                # Use Arabic language with Egyptian dialect
                recognized_text = recognizer.recognize_google(audio, language='ar-EG').lower()  # Convert to lowercase
                if recognized_text:
                    print("\rYou: " + recognized_text)
            except sr.UnknownValueError:
                recognized_text = ""
                responses = ["Can you repeat, SIR?", "I couldn't hear that, SIR.", "Sorry, SIR, I can't hear you.", "Sorry, SIR, can you repeat?"]
                speak(random.choice(responses))
                return listen()
            except sr.RequestError:
                responses = ["Can you repeat, SIR?", "I couldn't hear that, SIR.", "Sorry, SIR, I can't hear you.", "Sorry, SIR, can you repeat?"]
                speak(random.choice(responses))
                return listen()
            finally:
                print("\r", end='', flush=True)  # Erase "Listening...." and "Recognizing..."

            # Return the recognized text for further processing
            return recognized_text

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=32,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = IntentDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model_path = 'trained_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded trained model from disk.")
else:
    optimizer = AdamW(model.parameters(), lr=1e-5)

    def train_model(model, data_loader, optimizer, epochs=3):
        model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    train_model(model, train_loader, optimizer, epochs=3)

    torch.save(model.state_dict(), model_path)
    print("Model trained and saved.")

def predict_intent(command):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            command,
            add_special_tokens=True,
            max_length=32,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(output[0], dim=1)
        return label_encoder.inverse_transform(prediction.cpu().numpy())[0]

def generate_QR_code():
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=50,
                       corder=2)
    qr.add_data("https://www.youtube.com/")
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("hudhf.png")
    
    
    factory = qrcode.image.svg.SvgPathImage
    svg_img = qrcode.make

def searchYoutube(query):
    if "youtube" in query:
        speak("This is what I found for your search!") 
        query = query.replace("youtube search","")
        query = query.replace("youtube","")
        query = query.replace("jarvis","")
        web  = "https://www.youtube.com/results?search_query=" + query
        webbrowser.open(web)
        pywhatkit.playonyt(query)
        speak("Done, Sir")

def searchGoogle(query):
    if "google" in query:
        import wikipedia as googleScrap
        query = query.replace("jarvis","")
        query = query.replace("google search","")
        query = query.replace("google","")
        speak("This is what I found on google")

        try:
            pywhatkit.search(query)
            result = googleScrap.summary(query,1)
            speak(result)

        except:
            speak("No speakable output available")

def searchWikipedia(query):
    if "wikipedia" in query:
        speak("Searching from wikipedia....")
        query = query.replace("wikipedia","")
        query = query.replace("search wikipedia","")
        query = query.replace("jarvis","")
        results = wikipedia.summary(query,sentences = 2)
        speak("According to wikipedia..")
        print(results)
        speak(results)

def speak(text):
    OUTPUT_FILE = "output.mp3"
    loop = asyncio.get_event_loop()
    try:
        voice = 'en-US-JessaNeural'
        communicate = edge_tts.Communicate(text, voice)
        loop.run_until_complete(communicate.save(OUTPUT_FILE))
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(OUTPUT_FILE)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except pygame.error as e:
        print(f"Pygame error: {e}")
    finally:
        pygame.quit()


def username():
    global first_time
    if first_time:
        global city
        speak("Sir, could you please tell me the name of the city you're living in?")
        city = input("Enter your city: ").strip()  # إزالة المسافات الزائدة إن وجدت
        speak(f"Got it! You've set your city to {city}.")
        speak("What should I call you?")
        global uname
        uname = input()
        speak("And may I know your name, please?")
        uname = input("Enter your name: ").strip()
        speak(f"Nice to meet you, {uname}.")
        time.sleep(2)
        speak(f"Hello {uname}. It's a pleasure meeting you!")
        speak("My name is Eva, your personal AI voice assistant, designed and developed by Assem Sabry.")
        speak("I am here to help you. Just ask me, 'Eva, what can you do?'")
        first_time = False

def check_first_time():
    if not os.path.exists("first_time_flag.txt"):
        with open("first_time_flag.txt", "w") as flag_file:
            flag_file.write("Executed")
        first_time_function()

def first_time_function():
    speak("Hello,... I am your AI voice assistant.")
    speak("let's start with your name")
    username()


def get_current_time():
    # الحصول على الوقت الحالي بالتنسيق المناسب
    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M %p")  # 12-hour format with AM/PM
    return current_time

# مثال على شرط intent
predicted_intent = 'the time'  # قم بتغيير القيمة لاختبار الشروط

if predicted_intent == 'the time':
    current_time = get_current_time()
    speak(f"The current time is {current_time}")


def generate_QR_code():
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=50,
                       corder=2)
    qr.add_data = input("The link to make QR code : ")
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("QRcode.png")
    
    
    factory = qrcode.image.svg.SvgPathImage
    svg_img = qrcode.make

def get_city():
    global city  
    city = input("Please enter the city you live in: ")
    speak(f"You have set your city to {city}")

def get_weather():
    """جلب الطقس للمدينة المحددة."""
    search = f"temperature in {city}"
    url = f"https://www.google.com/search?q={search}"
    r = requests.get(url)
    data = BeautifulSoup(r.text, "html.parser")
    temp = data.find("div", class_="BNeawe").text
    speak(f"The current temperature in {city} is {temp}")

def main():
    check_first_time()  # Ensure this function is defined somewhere

    while True:
        command = listen()
        predicted_intent = predict_intent(command)
        speak(f"{predicted_intent}")

        if predicted_intent == 'weather':
            speak(f"Searhing for the weather in {city}")
            get_weather()

        elif predicted_intent == 'joke':
            responses = ["Sure", "Of course, Sir", "Yeah sure"]
            speak(random.choice(responses))
            joke = pyjokes.get_joke()
            speak(joke)

        elif predicted_intent == 'timer':
            speak("You have chosen to set a timer, sir")

        elif predicted_intent == 'open youtube':
            responses = ["Opening YouTube, Sir", "Give me a second, Sir", "Here you go to YouTube...Sir"]
            speak(random.choice(responses))
            webbrowser.open("https://www.youtube.com")

        elif predicted_intent == 'open google':
            responses = ["Opening Google, Sir", "Give me a second, Sir", "Here you go to Google...Sir"]
            speak(random.choice(responses))
            webbrowser.open("https://www.google.com")

        elif predicted_intent == 'restart computer':
            responses = ["Are you sure you want me to restart your PC..Sir?", "Waiting for your confirmation..Sir", "If you want me to restart your PC please confirm with Yes...Sir"]
            speak(random.choice(responses))
            confirm = listen()
            if confirm.lower() in ["yes", "yeah", "sure"]:
                subprocess.call(["shutdown", "/r"])
        
        elif predicted_intent == 'shutdown computer':
            responses = ["Are you sure you want me to shutdown your PC..Sir?", "Waiting for your confirmation..Sir", "If you want me to shutdown your PC please confirm with Yes...Sir"]
            speak(random.choice(responses))
            confirm = listen()
            if confirm.lower() in ["yes", "yeah", "sure"]:
                subprocess.call(['shutdown', '/p', '/f'])
        
        elif predicted_intent == 'sleep computer':
            responses = ["See you soon...Sir", "Of course...Sir", "Your PC is going into sleep mode...Sir"]
            speak(random.choice(responses))
            subprocess.call("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

        elif predicted_intent == 'lock computer':
            responses = ["Locking your PC...Sir", "Of course...Sir", "Well..Sir"]
            speak(random.choice(responses))
            ctypes.windll.user32.LockWorkStation()

        elif predicted_intent == 'owner':
            responses = [
                "Assem Sabry is an Egyptian programmer and he made me as one of his Artificial intelligence models to assist people and to be close to what humans can do",
                "I was made and developed by Assem Sabry, an Artificial intelligence developer",
                "My owner is Assem Sabry...he created me and named me Eva to assist people and make their life easier....As one of his models...Assem made me capable of doing many things to help you"
            ]
            speak(random.choice(responses))

        elif predicted_intent == 'pass words':
            speak("Done Sir")

        elif predicted_intent == 'mute':
            pyautogui.press('volumemute')

        elif predicted_intent == 'unmute':
            pyautogui.press('volumeup')
            time.sleep(2)

        elif predicted_intent == 'screenshot':
            speak("Sure, taking a screenshot.")
            im = pyautogui.screenshot()
            im.save(r"Desktop\screenshotEva.jpg")
            speak("Done, Sir, it's on your screen now.")

        elif predicted_intent == 'the time':
            current_time = get_current_time()
            speak(f"The current time is {current_time}")

        elif predicted_intent == 'hi assis':
            uname = "User"  # Replace with dynamic user name retrieval if needed
            responses = [f"Hello...{uname}", "Of course...Sir", "Well..Sir"]
            speak(random.choice(responses))

        elif 'empty recycle bin' in predicted_intent:
            winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=True)
            speak("Recycle Bin emptied.")

        elif "where is" in predicted_intent:
            location = predicted_intent.replace("where is", "").strip()
            speak("User asked to locate")
            speak(location)
            webbrowser.open(f"https://www.google.com/maps/place/{location}")

        elif "write a note" in predicted_intent:
            speak("What should I write, sir?")
            note = listen()
            with open('Tasks.txt', 'a') as file:
                speak("Sir, should I include date and time?")
                snfm = listen()
                if 'yes' in snfm.lower():
                    strTime = get_current_time()
                    file.write(f"{strTime} :- {note}\n")
                else:
                    file.write(f"{note}\n")

        elif "show note" in predicted_intent:
            speak("Showing Notes")
            with open("Tasks.txt", "r") as file:
                notes = file.read()
                print(notes)
                speak(notes)

if __name__ == "__main__":
    first_time = True
    main()