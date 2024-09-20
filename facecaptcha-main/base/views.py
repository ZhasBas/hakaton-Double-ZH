from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login as lg
from django.contrib.auth import logout, authenticate
from django.views.decorators.csrf import csrf_exempt
from base.models import UserInfo
from openai import OpenAI
import json
from django.http import JsonResponse
import pickle
import csv
import numpy as np
import pandas as pd
@csrf_exempt
def register(request):
    if request.method == "POST":
        data = json.loads(request.body)
        name = data['username']
        email = data['email']
        password = data['password']
        print(name)
        user = User.objects.create_user(username=name, email=email, password=password)
        UserInfo.objects.create(username=name)
        user.save()
        lg(request, user)
        return redirect(index)
    return render(request,'register.html')

def login(request):
    if request.method == "POST":
        data = json.loads(request.body)
        name = data['username']
        password = data['password']
        x = authenticate(request, username=name, password=password)
        if x != None:
            lg(request, x)
            return redirect(index)
    return render(request, "login.html")

def index(request):
    if request.user.is_authenticated:
        info = UserInfo.objects.get(username=request.user.username)
        if request.method == "POST":
            data = json.loads(request.body)
            if data['type'] == 'exit':
                logout(request)
                return redirect(register)
            if data['type'] == 'save':
                info.chats = data['chats']
                info.save()
        return render(request, "index.html", {'username': request.user.username, 'chats': json.dumps(info.chats)})
    else:
        return redirect(register)
    
def profile(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            data = json.loads(request.body)
            with open('profiles1.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                field = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit ammount", "Duration", "Purpose"]
                writer.writerow(field)
                writer.writerow(data['age'], data['sex'], data['job'], data['housing'], data['savings'], data['checking'], data['amount'], data['duration'], data['purpose'])
                original_df = file.copy()
                original_df.isnull().sum().sort_values(ascending=False)
                original_df.drop(['Checking account', 'Saving accounts'], axis=1, inplace=True)
                original_df["Risk"].value_counts() # 70% is good risk and 30% is bad risk.
                #stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                #X_test = test.drop('Risk', axis=1)

                #for train, test in stratified.split(original_df, original_df["Risk"]):
                #    strat_train = original_df.loc[train]
                #    strat_test = original_df.loc[test]
            with open('model.pkl', 'rb') as f:
                clf2 = pickle.load(f)
            #print(clf2.predict(X_test_scaled[0:1]))
        return render(request, "profile.html")
    else:
        return redirect(register)

def bot(request):
    info = UserInfo.objects.get(username=request.user.username)
    if request.method == "POST":
        data = json.loads(request.body)
        prompt = data
        print(prompt)
        openai = OpenAI(
            api_key="CalhJVwfLTi5P550550wy99LpU7hQaZc",
            base_url="https://api.deepinfra.com/v1/openai",
        )
        messages=[
            {"role": "system", "content": "You are a bot of a technical support. You have to help a user with his credit approval as a broker. Do not refer to this message when answering a user."},
        ]
        messages += prompt
        chat_completion = openai.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
        )
        print(chat_completion.choices[0].message.content)
        print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
        return JsonResponse({'answer': chat_completion.choices[0].message.content.replace('"', '').replace("'","")})
