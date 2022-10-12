import cv2
import os
import os.path as osp
import warnings
import mmcv
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result,
                         process_mmdet_results,
                         inference_bottom_up_pose_model)
from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import DatasetInfo
import matplotlib.pyplot as plt


# 定义可视化图像函数，输入图像路径，可视化图像
def show_img_from_path(img_path):
    '''opencv 读入图像，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# 定义可视化图像函数，输入图像 array，可视化图像
def show_img_from_array(img):
    '''输入 array，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()



# 目标检测模型
# det_config = '../demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 人体姿态估计模型
# pose_config = '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
# pose_checkpoint = '../hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
def main():
    pose_config = '../associative_embedding_hrnet_w32_coco_512x512.py'
    pose_checkpoint = '../hrnet_w32_coco_512x512-bcb8c247_20200816.pth'

# 目标检测模型
# det_model = init_detector(det_config, det_checkpoint)

# 人体姿态估计模型


    img_path = '../my_data'
    # prepare image list
    if osp.isfile(img_path):
        image_list = [img_path]
    elif osp.isdir(img_path):
        image_list = [
            osp.join(img_path, fn) for fn in os.listdir(img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
            ]
    else:
            raise ValueError('Image path should be an image or image folder.'
                             f'Got invalid image path: {img_path}')

    # show_img_from_path(img_path)
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    out_img_root = 'vis_results'
    for image_name in mmcv.track_iter_progress(image_list):

        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=0.9,
            return_heatmap=False,
            outputs=None)

        if out_img_root == '':
            out_file = None
        else:
            os.makedirs(out_img_root, exist_ok=True)
            out_file = os.path.join(
                out_img_root,
                f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')

        vis_pose_result(
                pose_model,
                image_name,
                pose_results,
                radius=4,
                thickness=2,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=0.3,
                show=False,
                out_file=out_file)

if __name__ == '__main__':
    main()