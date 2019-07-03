import runway
import numpy as np
import tensorflow as tf

from model import DeepLabModel

labels = [
  'background',
  'aeroplane',
  'bicycle',
  'bird',
  'boat',
  'bottle',
  'bus',
  'car',
  'cat',
  'chair',
  'cow',
  'diningtable',
  'dog',
  'horse',
  'motorbike',
  'person',
  'pottedplant',
  'sheep',
  'sofa',
  'train',
  'tvmonitor'
]

label_to_id = { label: i for i, label in enumerate(labels) }

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True)})
def setup(opts):
  return DeepLabModel(opts['checkpoint_dir'])


@runway.command('segment', inputs={'image': runway.image}, outputs={'segmentation': runway.segmentation(label_to_id=label_to_id, label_to_color={'background': [0,0,0]})})
def segment(model, inputs):
  _, seg_map = model.run(inputs['image'])
  return seg_map.astype(np.uint8)


@runway.command('mask', inputs={'image': runway.image}, outputs={'masked_image': runway.image})
def mask(model, inputs):
  _, seg_map = model.run(inputs['image'])
  mask = np.stack((seg_map,)*4, axis=-1)
  masked = np.array(inputs['image'].resize(seg_map.shape[::-1]))
  masked = np.dstack((masked, np.full(masked.shape[:-1], 255)))
  masked[mask != 15] = 0
  return masked.astype(np.uint8)


if __name__ == '__main__':
  runway.run(port=5232)