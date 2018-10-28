import sys

import os
from flask import Flask, request, abort, jsonify, Response
import json

import urllib.parse
from OpenSSL import crypto
import base64
import requests

import sys
sys.path.append('./vendor')
from linebot import LineBotApi

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, StickerSendMessage, ImageSendMessage
)
from linebot.exceptions import LineBotApiError
sys.path.append('../')


line_bot_api = LineBotApi('3e5qlbYb2Exgu4ZGGyJPS/SZPtWQTSPGCraGGQOA5zuzN9AAk6/eqd7Ej2DKFi1oYbEVvQembHE8C8eXgMS59Ox4mRoRVOJIo4/hpCWNt2dZY+ECwJIWZMEDXjN9IJ0IQ8L5MoeKy8fuQHvkiT2UzAdB04t89/1O/w1cDnyilFU=')


import urllib.request

def make_image_message():
    messages = ImageSendMessage(
        original_content_url="https://i.imgur.com/VHEWGaE.jpg", #JPEG 最大画像サイズ：240×240 最大ファイルサイズ：1MB(注意:仕様が変わっていた)
        preview_image_url="https://i.imgur.com/VHEWGaE.jpg" #JPEG 最大画像サイズ：1024×1024 最大ファイルサイズ：1MB(注意:仕様が変わっていた)
    )
    return messages

def get_ohira(get_name):
    print("http://192.168.179.6:3000/rooms/1/" + get_name)
    with urllib.request.urlopen("http://192.168.179.6:3000/rooms/1/" + get_name) as res:
       html = res.read().decode("utf-8")
    return html


app = Flask(__name__)
app.name_count = 0
app.previous_intent = None

@app.route('/', methods=['POST'])
def callback():
    #print(request.get_data())

    #if(validate_request(request.headers, request.get_data()) == False):
    #    abort(400)

    req = CekApi(json.loads(request.get_data(as_text=True)))

    body = request.get_data(as_text=True)
    event = json.loads(body)
    print(event)

    response = json.loads('{ \
        "version": "0.1.0", \
        "sessionAttributes": {}, \
        "response": { \
            "outputSpeech": { \
                "type": "SimpleSpeech", \
                "values": { \
                    "type": "PlainText", \
                    "lang": "ja", \
                    "value": "ここに返答" \
                } \
            }, \
            "reprompt": { \
                "outputSpeech": { \
                    "type": "SimpleSpeech", \
                    "values": { \
                        "type": "PlainText", \
                        "lang": "ja", \
                        "value": "" \
                    } \
                } \
            }, \
            "card": {}, \
            "directives": [], \
            "shouldEndSession": false \
            } \
        }')

    if event['request']['type'] == 'LaunchRequest':
        response['response']['outputSpeech']['values']['value'] = '見守っちゃうぞ'
    elif event['request']['type'] == 'IntentRequest':
        print(event['request']['intent']['name'])
        if event['request']['intent']['name'] == 'HelloIntent':
            response['response']['outputSpeech']['values']['value'] = 'ご挨拶して頂きありがとうございます。'
            response['sessionAttributes'] = {'lastIntent' : 'RegisterIntent'}
            app.previous_intent = "HelloIntent"

        elif event['request']['intent']['name'] == 'QuestionIntent':
            response['response']['outputSpeech']['values']['value'] = '好きな果物は何ですか？'
            app.previous_intent = "QuestionIntent"

        elif event['request']['intent']['name'] == 'AnswerIntent':
            answer = event['request']['intent']['slots']['fruits']['value']
            response['response']['outputSpeech']['values']['value'] = '%sがお好きなんですね。素敵です。' % answer
            app.previous_intent = "AnswerIntent"

        elif event['request']['intent']['name'] == 'Clova.GuideIntent':
            if app.previous_intent == 'NameIntent'  :
                response['response']['outputSpeech']['values']['value'] = '少々お待ちください。係の者が参ります'
                line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', TextSendMessage(text='不審者かもしれません。'))
                line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', make_image_message())
            response['response']['outputSpeech']['values']['value'] = ''
            app.previous_intent = "Clova.GuideIntent"

        elif event['request']['intent']['name'] == 'Clova.CancelIntent':
            response['response']['outputSpeech']['values']['value'] = '見守り終了。'
            response['response']['shouldEndSession'] = True
            app.previous_intent = "Clova.CancelIntent"

        elif event['request']['intent']['name'] == 'Clova.YesIntent':
            response['response']['outputSpeech']['values']['value'] = '承認のインテントです。'
            app.previous_intent = "Clova.YesIntent"

        elif event['request']['intent']['name'] == 'Clova.NoInte':
            response['response']['outputSpeech']['values']['value'] = '否定のインテントです。'
            app.previous_intent = "Clova.NoInte"


        elif event['request']['intent']['name'] == 'WhoAreIntent':
            ###
            #open_person= open('C:\\Users\\isowa\\Downloads\\python\\python\\test_name.json', 'r')
            #person_count = json.load(open_person) #ここが(2)
            person_count = json.loads(get_ohira("users.json"))
            print(person_count)
            try:
                answer = event['request']['intent']['slots']['worker_name']['value']
                if answer == '誰':
                    if len(person_count) != 0:
                        msg = ""
                        for ind in person_count:
                            msg += ind["name"] + ","
                            #msg += app.name_buffer[ind["name"]] + ","
                        response['response']['outputSpeech']['values']['value'] = '%sがいます' % msg
                    else :
                        response['response']['outputSpeech']['values']['value'] = '誰もいませんよ。外出、帰宅のさいは施錠をお願いしますね。'
                else:
                    check_person = False
                    for ind in person_count:
                        if ind["name"] == answer:
                            check_person = True
                            break
                    if check_person:
                        response['response']['outputSpeech']['values']['value'] = '%sはまだ残っています。' % answer
                    else:
                        response['response']['outputSpeech']['values']['value'] = '%sはいません。' % answer
            except KeyError:
                if len(person_count) != 0:
                    msg = ""
                    for ind in person_count:
                        msg += ind["name"] + ","
                        #msg += app.name_buffer[ind["name"]] + ","
                    response['response']['outputSpeech']['values']['value'] = '%sがいます' % msg
                else :
                    response['response']['outputSpeech']['values']['value'] = '誰もいませんよ。'
            app.previous_intent = "WhoAreIntent"

        elif event['request']['intent']['name'] == 'LoginNumberIntent':
            ###
            #open_number = open('C:\\Users\\isowa\\Downloads\\python\\python\\test.json', 'r') #ここが(1)
            #number_count = json.load(open_number) #ここが(2)
            number_count = json.loads(get_ohira("count"))
            print(number_count)

            response['response']['outputSpeech']['values']['value'] = '今%s人います。' % number_count['count']
            app.previous_intent = "LoginNumberIntent"

        elif event['request']['intent']['name'] == 'StrangerIntent':
            response['response']['outputSpeech']['values']['value'] = 'お名前は何ですか？'
            app.previous_intent = "StrangerIntent"

        elif event['request']['intent']['name'] == 'NameIntent'  :
            if app.previous_intent == "StrangerIntent":
                try:
                    answer = event['request']['intent']['slots']['worker_name']['value']
                    print(answer)
                    response['response']['outputSpeech']['values']['value'] = '%sさんですね、わかりました。' % answer
                    app.name_count = 0
                    app.previous_intent = "NameIntent"
                except KeyError:
                    app.name_count += 1
                    print(app.name_count)
                    if app.name_count < 1:
                        response['response']['outputSpeech']['values']['value'] = 'もう一度お願いします。'
                    else :
                        response['response']['outputSpeech']['values']['value'] = 'どなたにご用ですか？'
                        app.previous_intent = "NameIntent"
                        #try:

                        #except LineBotApiError as e:
                            #print('error')

                        app.name_count = 0
            else:
                response['response']['outputSpeech']['values']['value'] = 'もう一度お願いします。'

        elif event['request']['intent']['name'] == 'WorkIntent'  :
            if app.previous_intent == "NameIntent":
                try:
                    answer = event['request']['intent']['slots']['worker_name']['value']
                    print(answer)
                    response['response']['outputSpeech']['values']['value'] = '%sさんに御用ですね、わかりました。' % answer
                    line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', TextSendMessage(text='%sさんにお客さんです。' % answer))
                    line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', make_image_message())
                    app.previous_intent == "WorkIntent"
                except KeyError:
                    response['response']['outputSpeech']['values']['value'] = '少々お待ちください。係の者が参ります'
                    line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', TextSendMessage(text='誰に御用かわかりません。対応お願いします。'))
                    line_bot_api.push_message('U27380939fd4e86861e37c0e0f94a90ec', make_image_message())
                    app.previous_intent == "WorkIntent"

            else:
                response['response']['outputSpeech']['values']['value'] = 'もう一度お願いします。'


        elif event['request']['intent']['name'] == 'CheerIntent':
            ###
            #open_isLate_name = open('C:\\Users\\isowa\\Downloads\\python\\python\\test_isLate_name.json', 'r') #ここが(1)
            #isLate_count_name = json.load(open_isLate_name)
            isLate_count_name = json.loads(get_ohira("front_person"))
            print('get front person data')
            if len(isLate_count_name) != 0:
                person = isLate_count_name["name"]
                #open_isLate = open('C:\\Users\\isowa\\Downloads\\python\\python\\test_isLate.json', 'r') #ここが(1)
                ###
                #isLate_count = json.load(open_isLate)
                user_id = isLate_count_name["user_id"]
                print("http://192.168.179.6:3000/users/" + str(user_id) + "/is_late")
                with urllib.request.urlopen("http://192.168.179.6:3000/users/" + str(user_id) + "/is_late") as res:
                   html = res.read().decode("utf-8")
                   isLate_count = json.loads(html)

                if isLate_count["is_late"]:
                    response['response']['outputSpeech']['values']['value'] = '%sさん、お疲れ様です。遅くまでご苦労様です。' % person
                else:
                    response['response']['outputSpeech']['values']['value'] = '%sさん、お疲れ様です。今日は早いですね。' % person
            else:
                response['response']['outputSpeech']['values']['value'] = 'お疲れ様です。'

            app.previous_intent = "CheerIntent"


    return jsonify(response)

def validate_request(headers, body_raw):

    # validate cert url
    cert_url = headers.get('SignatureCEKCertChainUrl')
    if cert_url is None:
        return False
    parsed_url = urllib.parse.urlparse(cert_url)
    cert_validation_result = False
    if parsed_url.scheme == 'https':
        if parsed_url.hostname == "clova-cek-requests.line.me":
            cert_validation_result = True
    if cert_validation_result == False:
        return False

    # validate signature
    if headers.get('SignatureCEK') is None or headers.get('SignatureCEKCertChainUrl') is None:
        return False

    cert_str = requests.get(cert_url)
    certificate = crypto.load_certificate(crypto.FILETYPE_PEM, str(cert_str.text))
    if certificate.has_expired() is True:
        return False
    if certificate.get_subject().CN != "clova-cek-requests.line.me":
        return False
    decoded_signature = base64.b64decode(headers.get('SignatureCEK'))

    signature_validation_result = False

    if crypto.verify(certificate, decoded_signature, body_raw, 'sha1') is None:
        signature_validation_result = True

    if signature_validation_result == False:
        return False

    return True

class CekApi():
    def __init__(self, object):
        print()

if __name__ == '__main__':
    app.debug = True;
    app.run(host='0.0.0.0')
