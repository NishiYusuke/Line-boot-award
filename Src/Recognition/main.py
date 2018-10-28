# -*- coding: utf-8 -*-

import os,sys
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
import glob
import socket
import json
import urllib.request
from PIL import Image

slim = tf.contrib.slim

### ライブラリ使用部分 ###
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

VOC_LABELS = {
    0: 'none',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(len(VOC_LABELS))]

def write_bboxes(img, classes, scores, bboxes):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                 colors[cls_id],
                                 2)
            class_name = VOC_LABELS[cls_id]
            cv2.rectangle(img, (xmin, ymin-6), (xmin+180, ymin+6),
                                 colors[cls_id],
                                 -1)
            cv2.putText(img, '{:s} | {:.3f}'.format(class_name, score),
                           (xmin, ymin + 6),
                           cv2.FONT_HERSHEY_PLAIN, 1,
                           (255, 255, 255))
### ライブラリ使用部分 ###

### 自作部分 ###

## カスケード分類 ##
# トレーニング画像
train_path = 'Train'

# テスト画像
test_path = 'Test'

# Haar-like特徴分類器
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def registration(id):
    Face_label = ["A","B","C","D","E"]

    new_dir_path = 'Person/'+str(id)
    os.mkdir(new_dir_path)

    cam = cv2.VideoCapture(0)

    vidw = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vfile_flag = True # True:file is none / False: file is existed
    out = cv2.VideoWriter('./output.avi', int(fourcc), fps, (int(vidw), int(vidh)))

    cc = 0
    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        key = cv2.waitKey(1)
        # when ESC key is pressed break
        if key == 27:
            break

        img = capture.copy()
        dst = np.array(cv2.cvtColor(capture.copy(), cv2.COLOR_BGR2RGB))

        start = time.time()

        ### ライブラリ部分 ###
        rclasses, rscores, rbboxes =  process_image(img)
        write_bboxes(img, rclasses, rscores, rbboxes)

        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")#+"calss:"+str(rbboxes))

        ### ライブラリ部分 ###

        ### 自作部分 ###
        count = 0
        for bb in rbboxes:
            xs,ys,xe,ye = bb
            xs,xe = int(xs*vidh),int(xe*vidh)
            ys,ye = int(ys*vidw),int(ye*vidw)
            pil_img = Image.fromarray(dst[xs:xe,ys:ye])
            pil_img.save(new_dir_path+"/"+str(id)+str(cc)+'.png')
            #cv2.imwrite(str(count)+".png", dst[ys:ye,xs:xe])
            count += 1
        #cv2.imwrite("src.png", img)
        #cv2.imshow('SSD', img)
        cc += 1
        ### 自作部分 ###
    cam.release()
    if vfile_flag:
        out.release()
    cv2.destroyAllWindows()

def get_images_and_labels(path):
    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    # ファイル名を格納する配列
    files = []
    for _dir in ["A","B","C"]:
        file_path = glob.glob(os.path.join(path,_dir,"*.png"))
        for image_path in file_path:
            # 画像のパス
            #print(image_path.split('/')[2])

            # グレースケールで画像を読み込む
            image_pil = Image.open(image_path).convert('L')

            # NumPyの配列に格納
            image = np.array(image_pil, 'uint8')
            
            # Haar-like特徴分類器で顔を検知
            faces = faceCascade.detectMultiScale(image)

            # 検出した顔画像の処理
            for (x, y, w, h) in faces:
                # 顔を 200x200 サイズにリサイズ
                roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
                # 画像を配列に格納
                images.append(roi)
                f = image_path.split('/')[2]
                # ファイル名からラベルを取得
                labels.append(dic[f[0]])
                # ファイル名を配列に格納
                files.append(f)
    return images, labels, files

def find_post(Person_id,conf,id):
    url = "http://192.168.179.6:3000/rooms/"+str(id)+"/find/"
    #print(url)
    data = {
        "user_id":Person_id,#1,
        "confidence":conf,#0.4565
    }
    #print(json.dumps(data).encode())
    headers = {
        "Content-Type" : "application/json"
    }
    req = urllib.request.Request(url, json.dumps(data).encode("utf-8"), headers)
    with urllib.request.urlopen(req) as res:
        body = res.read()

if __name__ == '__main__':
    dic = {"A":1,"B":2,"C":3}
    # 指定されたpath内の画像を取得
    # トレーニング画像を取得
    images, labels, files = get_images_and_labels(train_path)

    # トレーニング実施
    recognizer.train(images, np.array(labels))

    # テスト画像を取得
    test_images, test_labels, test_files = get_images_and_labels(test_path)

    i = 0

    name = {1:"Ohira",2:"Isowa",3:"Shintani"}
    ### 自作部分 ###
    Face_label = ["1","2","3","4","5"]
    cc = 0

    # カメラ初期設定
    cam = cv2.VideoCapture(0)

    vidw = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    while True:
        # 画像の撮影
        ret, capture = cam.read()
        if not ret:
            print('error')
            break

        img = capture.copy()
        cv2.imwrite("src.png", img)
        dst = np.array(cv2.cvtColor(capture.copy(), cv2.COLOR_BGR2RGB))

        # オブジェクトの領域抽出
        rclasses, rscores, rbboxes =  process_image(img)
        write_bboxes(img, rclasses, rscores, rbboxes)

        start = time.time()
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")#+"calss:"+str(rbboxes))

        ### 自作部分 ###
        count = 0
        # 抽出した人間の領域(ボックス)毎に顔認識
        for bb in rbboxes:
            xs,ys,xe,ye = bb# ボックスの4点 
            xs,xe = int(xs*vidh),int(xe*vidh)
            ys,ye = int(ys*vidw),int(ye*vidw)

            # RGB 画像をグレースケールに変換
            pil_img = Image.fromarray(dst[xs:xe,ys:ye])
            pil_img.save(Face_label[count]+str(cc)+'.png')
            image_pil = Image.open(Face_label[count]+str(cc)+'.png').convert('L')
            image = np.array(image_pil, 'uint8')
            
            # Haar-like特徴分類器で顔を検知
            faces = faceCascade.detectMultiScale(image)

            # 検出した顔画像の処理
            for (x, y, w, h) in faces:
                # 顔を 200x200 サイズにリサイズ
                roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
                # 画像を配列に格納
                images.append(roi)
            # label:個人毎に割り振られたID score:分類に対する確信度
            label,score = 0,-100
            idx = 0
            for img in images:
                # 撮影した顔のパーツを入力として、予測
                _label, _con = recognizer.predict(img)
                # 予測結果をコンソール出力
                print("Test Image: {}, Confidence: {}".format(_label, _con))
                if  _con > score:
                    im = images[idx]
                    label = _label
                    score = _con
                idx += 1
            # HTTP サーバーへ結果を送信
            find_post(label,score,1)                        
            count += 1