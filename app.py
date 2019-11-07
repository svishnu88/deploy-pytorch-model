from flask import Flask,url_for,redirect,request
import requests
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
from service_streamer import ThreadedStreamer

torch.nn.SyncBatchNorm.convert_sync_batchnorm
import json

app = Flask(__name__)

def get_model():
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    model.eval()
    return model

def get_transforms():
    img_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return img_transforms

def get_transforms_batch(imgs):
    img_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    imgs = torch.cat([img_transforms()(im) for im in imgs])
    return imgs

model = get_model()

class ImagenetClassifier():
    def predict(self,urls):
        imgs = [get_img(url) for url in urls]
        imgs = get_transforms_batch(imgs)
        outp = model(imgs)
        return outp
        

img_transforms = get_transforms
imagenete_id_cat = json.load(open('imagenet_class_index.json'))

def get_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    return img

def get_preds(preds):
    idx = preds.max(1)[1].item()
    return imagenete_id_cat[str(idx)][1]

@app.route('/predict',methods=['POST'])
def predict():
    img =  get_img(request.args['url'])
    outp = model(img_transforms()(img).unsqueeze(0))
    return f'{get_preds(outp)}'

cmodel = ImagenetClassifier()
streamer = ThreadedStreamer(cmodel.predict,batch_size=64,max_latency=0.1)
@app.route("/stream", methods=["POST"])
def stream_predict():
    #img =  get_img(request.args['url'])
    outp = streamer.predict(request.args['url'])
    return outp

if __name__ == '__main__':
   app.run(debug=True)