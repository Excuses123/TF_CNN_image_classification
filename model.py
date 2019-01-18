import tensorflow as tf

class CNN(object):

    def __init__(self, config):
        self.Batch = config['BATCH_SIZE']
        self.C = config['N_CLASSES']
        self.W = config['IMAGE_WEIGHT']
        self.H = config['IMAGE_HEIGHT']
        self.lr = config['lr']

        self.X = tf.placeholder('float32',[self.Batch, self.W, self.H, 3])
        self.Y = tf.placeholder('int32',[self.Batch])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        #conv1.shape = [kernel_size,kernel_size,channel,kernel_number]
        with tf.variable_scope("conv1") as scope:
            weights = tf.get_variable("weights",
                                      shape=[3,3,3,16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(self.X,weights,strides=[1,1,1,1],padding="SAME")
            pre_activation = tf.nn.bias_add(conv,biases)
            conv1 = tf.nn.relu(pre_activation,name="conv1")

        #pool1 && norm1
        with tf.variable_scope("pooling1_lrn") as scope:
            pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pooling1")
            norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="norm1")

        #conv2
        with tf.variable_scope("conv2") as scope:
            weights = tf.get_variable("weights",
                                      shape=[3,3,16,16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding="SAME")
            pre_activation = tf.nn.bias_add(conv,biases)
            conv2 = tf.nn.relu(pre_activation,name="conv2")

        #pool2 && norm2
        with tf.variable_scope("pooling2_lrn") as scope:
            pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pooling2")
            norm2 = tf.nn.lrn(pool2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="norm2")

        #full-connect1
        with tf.variable_scope("fc1") as scope:
            reshape = tf.reshape(norm2,shape=[self.Batch,-1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable("weight",
                                      shape=[dim,128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape,weights) + biases,name="fc1")

            # # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

        #full-connect2
        with tf.variable_scope("fc2") as scope:
            weights = tf.get_variable("weights",
                                      shape=[128,128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.001,dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1,weights) + biases,name="fc2")

            # # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
            fc2 = tf.nn.dropout(fc2, self.keep_prob)

        #softmax
        with tf.variable_scope("softmax_linear") as scope:
            weights = tf.get_variable("weights",
                                      shape=[128,self.C],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
            biases = tf.get_variable("biases",
                                     shape=[self.C],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))
            pred = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
            self.logits = tf.nn.softmax(pred)


    def losses(self):
        with tf.variable_scope("loss") as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y,name="xentropy_per_example")
            self.train_loss = tf.reduce_mean(cross_entropy,name="loss")
            tf.summary.scalar(scope.name + "loss",self.train_loss)

    def training(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            global_step = tf.Variable(0,name="global_step",trainable=False)
            self.train_op = optimizer.minimize(self.train_loss,global_step=global_step)


    def evaluate(self):
        with tf.variable_scope("accuracy") as scope:
            correct = tf.nn.in_top_k(self.logits,self.Y,1)
            correct = tf.cast(correct,tf.float16)
            self.train_acc = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + "accuracy",self.train_acc)

    def bulid_graph(self):
        self.inference()
        self.losses()
        self.evaluate()
        self.training()
















