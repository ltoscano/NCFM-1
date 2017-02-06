import json
from PIL import Image
import tensorflow as tf

def getBoxes(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)

alb_boxes = getBoxes("boxes/alb_labels.json")
bet_boxes = getBoxes("boxes/bet_labels.json")
dol_boxes = getBoxes("boxes/dol_labels.json")
lag_boxes = getBoxes("boxes/lag_labels.json")
shark_boxes = getBoxes("boxes/shark_labels.json")
yft_boxes = getBoxes("boxes/yft_labels.json")

#print data[0]['filename']
#print data[0]['annotations'][0]['x'], data[0]['annotations'][0]['y']


def saveCropped(boxes, species, pathfuckedup=False):
    for box in boxes:
        if pathfuckedup:
            origpath = "train/" + species + "/" + box['filename'].split("/")[-1]
        else:
            origpath = "train/" + species + "/" + box['filename']
        origname = origpath.split('/')[-1]
        #print origpath
        orig = Image.open(origpath)
        for i, ann in enumerate(box['annotations']):
            x = int(ann['x'])
            y = int(ann['y'])
            w = int(ann['width'])
            h = int(ann['height'])
            #print orig.size
            print float(orig.size[0]) / orig.size[1]
            #print x, y, w, h
            #fish = orig.crop((x, y, x+w, y+h))
            #path = "cropped_train/" + species + "/" + origname.split(".")[0] + "-" + str(i) + "." + origname.split(".")[1]
            #print path
            #fish.save(path, "JPEG")


# Do it
saveCropped(alb_boxes, "ALB")
saveCropped(bet_boxes, "BET")
saveCropped(dol_boxes, "DOL")
saveCropped(lag_boxes, "LAG")
saveCropped(shark_boxes, "SHARK", True)
saveCropped(yft_boxes, "YFT", True)
