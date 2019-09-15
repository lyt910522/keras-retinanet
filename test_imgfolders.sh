#!/bin/bash
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_A ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_B ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_C ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_D ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_E ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_F ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/imagesfrom_Scene_G ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneA_Rotated_Left ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneA_Rotated_Right ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneB_Rotated_Left ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneB_Rotated_Right ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneC_Rotated_Left ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneC_Rotated_Right ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneD_Rotated_Left ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneD_Rotated_Right ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneE_Rotated_Left ./test_resnet50_5cls.h5 --json
python keras_retinanet/bin/predict.py ~/workspace/datasets/microcam/test_images/images/SceneE_Rotated_Right ./test_resnet50_5cls.h5 --json

