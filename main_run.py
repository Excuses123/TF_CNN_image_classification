from data_helper import *
from model import CNN


if __name__ == '__main__':
    train_dir = "./data/train/"
    # 存放一些模型文件的目录
    log_train_dir = "./log/"

    config = {}
    # 要分类的类别数，这里是2分类
    config['N_CLASSES'] = 2
    # 设置图片的size
    config['IMAGE_WEIGHT'] = 256
    config['IMAGE_HEIGHT'] = 256
    config['BATCH_SIZE'] = 12
    # 学习率
    config['lr'] = 0.0001
    # 迭代一千次，如果机器配置好的话，建议至少10000次以上
    MAX_STEP = 2000

    image_list, label_list = get_files(train_dir)

    model = CNN(config)
    model.bulid_graph()

    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_train_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())

    # tf.summary.FileWriter('improved_graph2', sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            train_batch, train_label_batch = next_batch(image_list, label_list, config['IMAGE_WEIGHT'], config['IMAGE_HEIGHT'], config['BATCH_SIZE'])
            feed_dict = {model.X:train_batch, model.Y:train_label_batch, model.keep_prob:0.5}
            _,tra_loss,tra_acc = sess.run([model.train_op, model.train_loss, model.train_acc], feed_dict = feed_dict)
            if step % 50 == 0:
                # 每迭代50次，打印出一次结果
                print('Step %d,train loss = %.5f,train accuracy = %.5f' % (step, tra_loss, tra_acc))
                # summary_str = sess.run(summary_op)
                # train_writer.add_summary(summary_str,step)

            if step % 1000 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(log_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
                # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()










