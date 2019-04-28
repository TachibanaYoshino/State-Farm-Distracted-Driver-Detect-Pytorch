from easydict import EasyDict as edict


__C = edict()

cfg = __C

# Train options
__C.TRAIN = edict()


__C.TRAIN.NUM_CLASSES = 10

__C.TRAIN.LEARNING_RATE = 0.0001

__C.TRAIN.MAX_EPOCHS = 100

__C.TRAIN.use_gpu = True

__C.TRAIN.BATCH_SIZE = 32 # res18,res32,res50,res101 : 32   res152 : 24

__C.TRAIN.frequency_print = 100

__C.TRAIN.train_data_path = './data/train'





# Test options
__C.TEST = edict()

__C.TEST.test_data_path = './data/test'

__C.TEST.BATCH_SIZE = 1

__C.TEST.use_gpu = True