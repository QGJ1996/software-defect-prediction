import yaml

f = open("./config.yaml",'r',encoding="utf-8")

string = f.read()


param_dic = yaml.load(string,Loader=yaml.FullLoader)

print(param_dic)