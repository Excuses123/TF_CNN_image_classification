import os
import numpy as np
from PIL import Image
from model import *
import matplotlib.pyplot as plt
import random

def get_files(file_dir):
    #分两类
    train0 = []
    label0 = []

    train1 = []
    label1 = []

    for file in os.listdir(file_dir):
        names = file.split("_")
        if names[0] == "0":
            train0.append(file_dir + file)
            label0.append(0)
        else:
            train1.append(file_dir + file)
            label1.append(1)
    print("There are %d train0 \n There are %d train1"%(len(train0),len(train1)))

    image_list = np.hstack((train0,train1))
    label_list = np.hstack((label0,label1))

    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]

    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    #tf.cast用来做数据转换

    input_queue = tf.train.slice_input_producer([image,label])
    #加入队列

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    #resize

    image = tf.image.per_image_standardization(image)
    #对resize后的图片进行标准化处理

    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=16,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch
    #获取两个batch，两个batch即为传入神经网络的数据


def get_one_image(img_dir,W,H):
    image = Image.open(img_dir)
    # Image.open()
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    plt.imshow(image)
    image = image.resize([W, H])
    image_arr = np.array(image)
    return image_arr

def next_batch(image_list, label_list,IMAGE_WEIGHT,IMAGE_HEIGHT, batch_size):
    index = [ i for i in range(0,len(label_list)) ]
    np.random.shuffle(index);
    batch_data = [];
    batch_target = [];
    for i in range(0,batch_size):
        image_arr = get_one_image(image_list[index[i]], IMAGE_WEIGHT, IMAGE_HEIGHT)
        batch_data.append(image_arr)
        batch_target.append(label_list[index[i]])

    return np.array(batch_data), batch_target


def test(model,test_file):
    log_dir = "./log/"
    image_arr = get_one_image(test_file,512,512)
    image_arr = image_arr.reshape([1,512,512,3])
    res = ["猫","狗"]

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess,ckpt.model_checkpoint_path)
            #调用saver.restore()函数,加载训练好的网络模型

            print("Loading success")
        else:
            print("No checkpoint")
        prediction = sess.run(model.logits,feed_dict={model.X: image_arr, model.keep_prob:1})
        max_index = np.argmax(prediction)
        print('预测的标签为：',max_index)
        print('预测的概率为：',prediction)
        print('预测的结果为：',res[max_index])

    if max_index == 0:
        print('This is a cat with possibility %.6f' % prediction[:, 0])
    elif max_index == 1:
        print('This is a dog with possibility %.6f' % prediction[:, 1])





