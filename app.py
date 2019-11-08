from flask import Flask,url_for,redirect,request
import requests
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
from service_streamer import ThreadedStreamer
import json
import pdb

app = Flask(__name__)
device = torch.device('cuda')
def get_model():
    model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
    model = model.to(device)
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
    img_t = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    #pdb.set_trace()
    imgs = torch.cat([img_t(img).unsqueeze(0) for img in imgs])
    return imgs

model = get_model()


def predict_batch(urls):
    imgs =  [get_img(url) for url in urls]
    imgs = get_transforms_batch(imgs)
    imgs = imgs.to(device)
    outp = model(imgs)
    return [get_preds(outp)]
        

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
    img_t = img_transforms()(img).unsqueeze(0).to(device)
    outp = model(img_t)
    return f'{get_preds(outp)}'

streamer = ThreadedStreamer(predict_batch,batch_size=64,max_latency=0.1)
@app.route("/stream", methods=["POST"])
def stream_predict():
    url = request.args['url']
    outp = streamer.predict([url])[0]
    return str(outp)

if __name__ == '__main__':
   app.run(debug=False)