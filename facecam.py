import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils import get_config
from trainer import Trainer
import argparse
import pickle
import cv2
from imutils.video import VideoStream
import time
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--embedding','-e',type=str,default=None,help='path to pickle file for target style.')
parser.add_argument('--class-folder','-cf',type=str,default=None,help='Path to the folder containing class images.')
# TODO
# add camera index
# add rotated mode
# add aspect ratio adjustment

INPUT_SIZE = 128

config = get_config('configs/funit_animals.yaml')
config['batch_size'] = 1
print('\033[32m'+'\t* Configuring the model\033[0m')
trainer = Trainer(config)
print('\033[32m'+'\t* Loading the weights\033[0m')
trainer.load_ckpt('pretrained/animal149_gen.pt')
trainer.eval()
transform_list = [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))] + transform_list
transform = transforms.Compose(transform_list)
face_detector = cv2.CascadeClassifier('./files/haar_cascade_face.xml')


def main(embedding):
    vs = VideoStream(src = 0, resolution= (1280,960),framerate=32).start()
    print('\033[32m'+'\t* Preparing for streaming\033[0m')
    time.sleep(1.0)
    frame = vs.read()
    mask = create_mask(INPUT_SIZE)
    while True:
        t0 = time.perf_counter() 
        frame = vs.read()
        frame = cv2.flip(frame,1)
        box = detect_faces(frame)
        if box is not None:
            # extend the size of bounding box
            box = extend_box(box,frame.shape[:2],0.8)
            
            # extract subject from image
            face_original = frame[box[0]:box[1],box[2]:box[3],:]

            # pass the subject through the model
            h,w,_ = face_original.shape
            face = Image.fromarray(face_original)
            face = convert_image(face,embedding)
            face = np.array(face)

            # applying mask to the output
            face = apply_mask(face[:,:,::-1], cv2.resize(face_original,(INPUT_SIZE,INPUT_SIZE)),mask )
            # resize model's output to match the initial size of subject
            face = cv2.resize(face,(w,h))

            # replacing the output back into the image
            frame[box[0]:box[1],box[2]:box[3],:] = face
        cv2.imshow('Output',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('\033[32m'+'\n\t* Process Terminated by user.\033[0m')
            break
        print('\033[1m'+f'\t{1/(time.perf_counter()-t0):.2f} fps'+'\033[0m',end='\r',flush=True)

def extract_embedding(path):
    print('Compute average class codes for images in '+ path)
    images = os.listdir(path)
    for i, f in enumerate(images):
        fn = os.path.join(path, f)
        img = Image.open(fn).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            class_code = trainer.model.compute_k_style(img_tensor, 1)
            if i == 0:
                new_class_code = class_code
            else:
                new_class_code += class_code
    final_class_code = new_class_code / len(images)
    print(final_class_code.shape)
    return final_class_code

def load_embedding(path):
    with open(path,'rb') as fp:
        embedding = pickle.load(fp)
    return embedding

def create_mask(size):
    mask = np.zeros((size,size),dtype = np.uint8)
    radius = size//2 - size//20
    mask = cv2.circle(mask,(size//2,size//2),radius=radius,color=(255),thickness = -1)
    ksize = size//12 *2 - 1
    mask = cv2.GaussianBlur(mask, (ksize,ksize),size//5)
    return mask

def apply_mask(img1,img2,mask):
    mask = mask /255.0
    output = img1*mask[...,None] + img2*(1-mask[...,None])
    return output.astype(np.uint8)

def convert_image(image,embedding):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_image = trainer.model.translate_simple(image, embedding)
        image = output_image.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
    return output_img

def detect_faces(image):
    # pdb.set_trace()
    # height, width , _ =image.shape
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = [(y,y+h,x,x+w) for (x, y, w, h) in rects]
    if len(boxes)>0:
        return boxes[0]
    else:
        return None

def extend_box(box,image_size,extend = 0.5):
    ymin,ymax,xmin,xmax = box
    img_h, img_w = image_size
    w = xmax - xmin
    h = ymax - ymin
    dy = int(h*extend//2)
    dx = int(w*extend//2)
    ymin -= dy
    ymax += dy
    xmin -= dx
    xmax += dx
    ymin = np.clip(ymin,0,img_h)
    ymax = np.clip(ymax,0,img_h)
    xmin = np.clip(xmin,0,img_w)
    xmax = np.clip(xmax,0,img_w)
    return (ymin, ymax, xmin, xmax)



if __name__=='__main__':
    args = parser.parse_args()

    if args.class_folder is not None:
        embedding = extract_embedding(args.class_folder)
    elif args.embedding is not None:
        embedding = load_embedding(args.embedding)
    else:
        raise Exception("No embedding info was passed in.")

    main(embedding)





