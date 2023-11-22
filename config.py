from model import ModelType

class Config:
    MODEL_TYPE = ModelType.LinearNet

    # 模型路径及是否预训练
    REG_PATH = './checkpoint/reg.pkl'
    SAVE_MODEL_PATH = './checkpoint/model.pth'
    SAVE_FINAL_MODEL_PATH = './checkpoint/final_model.pth'
    PRETRAINED = False

    # 数据集路径
    DATASET_PATH = '../train.csv'
    TEST_DATASET_PATH = '../test.csv'
    TEST_DATASET_LABEL_PATH = '../test_label.csv'
    SAVE_RESULT_PATH = './result.csv'

    # 训练参数
    EPOCHS = 12
    BATCH_SIZE = 64
    LEARNING_RATE = 0.002
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    STEP_LR_STEP_SIZE = 5
    STEP_LR_GAMMA = 0.5

    RANDOM_SEED = 10