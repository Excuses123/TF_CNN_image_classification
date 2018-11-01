from data_helper import *
from model import *


N_CLASSES = 2
#要分类的类别数，这里是2分类
IMAGE_WEIGHT = 208
IMAGE_HEIGHT = 208
#设置图片的size
BATCH_SIZE = 8
CAPACITY = 64
MAX_STEP = 1000
#迭代一千次，如果机器配置好的话，建议至少10000次以上
learning_rate = 0.0001
#学习率
KEEP_PROB = 0.5


if __name__ == '__main__':

    train_dir = "./data/train/"
    log_train_dir = "./log/"
    #存放一些模型文件的目录

    train,train_label = get_files(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,IMAGE_WEIGHT,IMAGE_HEIGHT,BATCH_SIZE,CAPACITY)

    train_logits = inference(train_batch,BATCH_SIZE,N_CLASSES,KEEP_PROB)
    train_loss = losses(train_logits,train_label_batch)
    train_op = training(train_loss,learning_rate)
    train_acc = evaluate(train_logits,train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(log_train_dir,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            if step % 50 == 0:
                print('Step %d,train loss = %.5f,train accuracy = %.5f' % (step, tra_loss, tra_acc))
                # 每迭代50次，打印出一次结果
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)

            if step % 200 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(log_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
                # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()










