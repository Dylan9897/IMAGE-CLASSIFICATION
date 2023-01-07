import json
import configparser



# 从配置文件中读取相关参数
config = configparser.ConfigParser()
config.read("config/config.ini", encoding="utf-8")

def parserConfig(args):
    param = {}
    model_name = args.model
    param['model'] = model_name
    param["trainset"] = config.get("DataSet","train")
    param["testset"] = config.get("DataSet","test")
    param["ckpt"] = config.get("ModelPath","root")
    param["log"] = config.get("LogPath","root")
    param["params"] = eval(config.get(model_name,"parameters"))
    return param

if __name__ == "__main__":
    res = config.get("GoogleNet","parameters")
    res = eval(res)
    print(type(res))

