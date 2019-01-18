from data_helper import *
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 5
CAPACITY = 64
IMAGE_WEIGHT = 1024
IMAGE_HEIGHT = 1024

train_dir = "./data/train/"

image_list,label_list = get_files(train_dir)
image_batch,label_batch = get_batch(image_list,label_list,IMAGE_WEIGHT,IMAGE_HEIGHT,BATCH_SIZE,CAPACITY)


with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    try:
        while not coord.should_stop() and i < 2:
            #提取出两个batch的图片并可视化
            img,label = sess.run([image_batch,label_batch])

            for j in np.arange(BATCH_SIZE):
                print('label: %d'%label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
        print("all threads are asked to stop")
    coord.join(threads)
    print("all threads are stopped")









