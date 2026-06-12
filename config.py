from model import Config

class attn_gin_tf(Config):
    classifier = "MClassifier"
    data_source = "drugbank"
    split_type = "random"
    epochs = 200
    node_dim = 86
    edge_dim = 13
    graph_dim = 15
    d_model = 128
    lr = 0.001
    heads = 8
    dp_r = 0.1
    train_size = 0.8
    seed = 42
    block_num = 8
    class_num = 86
    drug_batch_size = 2048
    itc_batch_size = 20480
    label_smoothing = 0.1
    weight_decay = 5e-4

