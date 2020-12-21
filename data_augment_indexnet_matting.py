import torchvision.transforms as transforms 
from tqdm import tqdm
import torchvision 
import numpy as np
import torch 
import cv2 
import os 



class DataAugmentation:
    def __init__(self,data_root_dir="../dataset/",save_images_dir="indexnet_matting/examples/images/",save_trimaps_dir="indexnet_matting/examples/trimaps/"):
        try:
            os.makedirs(save_images_dir)
            os.makedirs(save_trimaps_dir)
        except FileExistsError:
            pass 

        for name in os.listdir(data_root_dir):
            for f in tqdm(os.listdir(data_root_dir+name+"/")):
                path        =   data_root_dir+name+"/"+f
                img,trimap  =   self.__maesyori(path)  
                cv2.imwrite(save_images_dir+name+"_"+f,img)   
                cv2.imwrite(save_trimaps_dir+name+"_"+f,trimap)

    def __maesyori(self,img_path):
        original_img    =   cv2.imread(img_path)
        h,w,_   =   original_img.shape 
        img     =   cv2.resize(original_img,(320,320))

        device  =   torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model   =   torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model   =   model.to(device)
        model.eval()

        preprocess= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.4556,0.406],std=[0.229, 0.224, 0.225]),
        ]) 
        input_tensor=   preprocess(img)
        input_batch =   input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output  =   model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        mask_array = output_predictions.byte().cpu().numpy()
        mask_array  =   mask_array*255
        mask        =   cv2.resize(mask_array,(w,h))
        kernel      =   np.ones((25,25),np.uint8)
        erosion     =   cv2.erode(mask,kernel,iterations=1)
        dilation    =   cv2.dilate(mask,kernel,iterations=1)
        trimap      =   cv2.addWeighted(dilation,0.5,erosion,0.5,0)
        return original_img,trimap 

if __name__ == "__main__":
    da  =   DataAugmentation()
    pass 
