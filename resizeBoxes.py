import json
from PIL import Image

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


def saveResized(boxes, species, pathfuckedup=False):
    for box in boxes:
        if pathfuckedup:
            origpath = "train/" + species + "/" + box['filename'].split("/")[-1]
        else:
            origpath = "train/" + species + "/" + box['filename']
        origname = origpath.split('/')[-1]
        print origpath
        orig = Image.open(origpath)
        for i, ann in enumerate(box['annotations']):
            print orig.size
            xratio = orig.size[0] / 1280.0
            yratio = orig.size[1] / 720.0
            ann['x'] = ann['x'] / xratio
            ann['y'] = ann['y'] / yratio
            ann['width'] = ann['width'] / xratio
            ann['height'] = ann['height'] / yratio
        fish = orig.resize((1280, 720))
        path = "1280720/" + species + "/" + origname
        print path
        #fish.save(path, "JPEG")

    return boxes


# Do it
new_box = saveResized(alb_boxes, "ALB")
with open('1280720boxes/alb_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
new_box = saveResized(bet_boxes, "BET")
with open('1280720boxes/bet_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
new_box = saveResized(dol_boxes, "DOL")
with open('1280720boxes/dol_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
new_box = saveResized(lag_boxes, "LAG")
with open('1280720boxes/lag_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
new_box = saveResized(shark_boxes, "SHARK", True)
with open('1280720boxes/shark_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
new_box = saveResized(yft_boxes, "YFT", True)
with open('1280720boxes/yft_labels.json', 'w') as new_box_path:
    json.dump(new_box, new_box_path)
