'''predict from images'''
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
import json

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.predict_image_generator import PredictImageGenerator
from ..utils.visualization import draw_detections
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version

import pdb


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    validation_generator = PredictImageGenerator(
        args.imgDir,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
    )
    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('imgDir',            help='Path to image folder.')
    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--json',             help='store json detection output.', action='store_true')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.29, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # print model summary
    # print(model.summary())

    if args.json:
        detection_output = []
        ori_path = args.imgDir
        json_file_name = os.path.join('.', ori_path.replace('/', ' ').strip().split(' ')[-1]+'.json')

    # start prediction
    for i in tqdm(range(generator.size())):
        raw_image    = generator.load_image(i)
        image, scale = generator.resize_image(raw_image.copy())
        image_file_name = generator.get_file_name(i)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > args.score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:args.max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]

        if args.json:
            n_output = len(image_scores)
            temp_output = {}
            temp_output['file_name'] = image_file_name
            temp_output['bbox'] = []
            temp_output['score'] = []
            temp_output['label'] = []
            for temp_box, temp_score, temp_label in zip(image_boxes, image_scores, image_labels):
                if temp_label == 0:
                    temp_output['bbox'].append(str(temp_box.tolist()))
                    temp_output['score'].append(str(temp_score))
                    temp_output['label'].append(generator.label_to_name(temp_label))
            detection_output.append(temp_output)

        if args.save_path is not None:
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=args.score_threshold)
            cv2.imwrite(os.path.join(args.save_path, '{}.png'.format(i)), raw_image)

    if args.json:
        with open(json_file_name, 'w') as fout:
            json.dump(detection_output, fout, indent=2)
    


if __name__ == '__main__':
    main()
