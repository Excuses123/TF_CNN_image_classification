from data_helper import *
config = {}
# 要分类的类别数，这里是2分类
config['N_CLASSES'] = 2
# 设置图片的size
config['IMAGE_WEIGHT'] = 256
config['IMAGE_HEIGHT'] = 256
config['BATCH_SIZE'] = 1
# 学习率
config['lr'] = 0.0001
# 迭代一千次，如果机器配置好的话，建议至少10000次以上
model = CNN(config)
model.bulid_graph()

test(model, "./data/test/54c8dd63-049b-465c-8d78-d761ac6bcdda.jpg")  #狗  1
test(model, "./data/test/92f58c79-dc50-49ba-a75a-9978ddac877e.jpg")  #狗  1
test(model, "./data/test/98c71123-8ea8-45f6-b8ab-4f6b28d2f7bc.jpg")  #狗  1
test(model, "./data/test/98e59dc5-eb5a-4ade-a2e5-834f1710ac5f.jpg")  #狗  1
test(model, "./data/test/gou1.jpg") #狗  1


print("mao==================================")
test(model, "./data/test/61b23673-2483-4bdd-8b8a-67c6876372d9.jpg")  #猫  0
test(model, "./data/test/62ab4841-1cf3-4fd2-880d-a7b95d1c5c69.jpg")  #猫  0
test(model, "./data/test/97a7f4fb-3562-40d7-8875-cf5971606a62.jpg")  #猫  0
test(model, "./data/test/97adebfe-afdf-4ebc-aa79-94ddf298674d.jpg")  #猫  0

test(model, "./data/test/timg.jpeg")  #猫  0
test(model, "./data/test/timg (1).jpeg")  #猫  0
test(model, "./data/test/mao3.jpg") #猫  0

test(model, "./data/test/mao_0.jpg")  #猫  0
test(model, "./data/test/tx.jpg")  #猫  0
test(model, "./data/test/touxiang.jpg") #猫  0



