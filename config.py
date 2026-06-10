from typing import ClassVar, Literal


class Config:
    _required_fields = {
        "encoder",
        "data_source",
        "split_type",
        "epochs",
        "node_dim",
        "edge_dim",
        "graph_dim",
        "d_model",
        "lr",
        "heads",
        "dp_r",
        "train_size",
        "seed",
        "block_num",
        "class_num",
        "drug_batch_size",
        "itc_batch_size",
        "label_smoothing",
    }
    encoder: ClassVar[Literal["AttnGINTFEncoder", "GINEncoder"]]
    data_source: ClassVar[Literal["drugbank", "twosides"]]
    split_type: ClassVar[Literal["random", "cluster"]]
    epochs: ClassVar[int]
    node_dim: ClassVar[int]
    edge_dim: ClassVar[int | None]
    graph_dim: ClassVar[int | None]
    d_model: ClassVar[int]
    lr: ClassVar[float]
    heads: ClassVar[int | None]
    dp_r: ClassVar[float]
    train_size: ClassVar[float]
    seed: ClassVar[int]
    block_num: ClassVar[int]
    class_num: ClassVar[int]
    drug_batch_size: ClassVar[int]
    itc_batch_size: ClassVar[int]
    label_smoothing: ClassVar[float]

    @classmethod
    def __init_subclass__(cls):
        for field in cls._required_fields:
            if field not in cls.__dict__:
                raise NotImplementedError(
                    f"子类 {cls.__name__} 必须显式设置属性: {field}"
                )


class attn_gin_tf_de(Config):
    encoder = "AttnGINTFEncoder"
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
    block_num = 5
    class_num = 86
    drug_batch_size = 2048
    itc_batch_size = 20480
    label_smoothing = 0.1


class gin(Config):
    encoder = "GINEncoder"
    data_source = "drugbank"
    split_type = "random"
    epochs = 200
    node_dim = 86
    edge_dim = None
    graph_dim = None
    d_model = 128
    lr = 0.001
    heads = None
    dp_r = 0.1
    train_size = 0.8
    seed = 42
    block_num = 5
    class_num = 86
    drug_batch_size = 2048
    itc_batch_size = 20480
    label_smoothing = 0.1
