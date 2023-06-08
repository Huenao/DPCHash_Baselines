def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_10":
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 24

    if "cifar" in config["dataset"]:
        config["data_path"] = "./dataset/cifar/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "nuswide_10":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/coco/"
    if config["dataset"] == "mirflickr":
        config["data_path"] = "./dataset/flickr25k/mirflickr/"
    config["data_list"] = {
        "train_dataset": "./data/" + config["dataset"] + "/train.txt",
        "test_dataset": "./data/" + config["dataset"] + "/test.txt",
        "database_dataset": "./data/" + config["dataset"] + "/database.txt"}
    return config