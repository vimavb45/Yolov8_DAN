import os
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torchvision.models as models
from ultralytics import YOLO
from torchvision.models import resnet50
from torch.fx import symbolic_trace

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #m = resnet50()
    yolo = YOLO('yolov8m-seg.pt')#._named_members
    #ckpt = torch.load('yolov8m-seg.pt')
    #print(ckpt['model'])
    #print(yolo.named_parameters())
    #model = yolov8
    #print(model)
    #for k, v in yolo.named_parameters():
    #    print(k, v)
    #train_nodes, eval_nodes = get_graph_node_names(model)
    #print(train_nodes)
    #print(eval_nodes)
    #print(model)
    #print(yolo.modules)
    return_nodes = {
    'layer7': 'layer7'
    }
    #for name, param in yolo.named_parameters():
    #    if name == 'yolo.model.7.conv.weight':
    #        print(name)
    #        print(param.size())

    #feature_tensor = yolo.model.model[7].forward('https://ultralytics.com/images/bus.jpg')

    #feature_tensor = yolo.model.model[0]('https://ultralytics.com/images/bus.jpg')
    #results = yolo('https://ultralytics.com/images/bus.jpg')  # predict on an image
    #print(feature_tensor)
    
    #print(model)
    #symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
    #print(symbolic_traced.graph)
    #results = model.train(data='Guardrail_dataset_56.yaml', model ='yolov8l-seg.pt', epochs=10, imgsz=640, batch=32, device=0)
    #print(create_feature_extractor(model, return_nodes=return_nodes))
    
    results = yolo.train(data='C:/Users/vvebh/OneDrive - Danmarks Tekniske Universitet/Skrivebord/PhD/VS Code/YOLOv8_V4/Guardrail_dataset_56.yaml', epochs=20000, imgsz=640, batch=16, device=0)
