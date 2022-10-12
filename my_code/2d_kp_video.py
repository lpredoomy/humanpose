import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

def video_kp(video_path, config, checkpoint, out_video_root):
    pose_model = init_pose_model(config, checkpoint)

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

        # read video
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {video_path}'

        if out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            size = (video.width, video.height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(out_video_root,
                             f'vis_{os.path.basename(video_path)}'), fourcc,
                fps, size)
            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None

            print('Running inference...')
            for _, cur_frame in enumerate(mmcv.track_iter_progress(video)):
                pose_results, _ = inference_bottom_up_pose_model(
                    pose_model,
                    cur_frame,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    pose_nms_thr=0.9,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names)

                # show the results
                vis_frame = vis_pose_result(
                    pose_model,
                    cur_frame,
                    pose_results,
                    radius=5,
                    thickness=2,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=0.3,
                    show=False)

                if save_out_video:
                    videoWriter.write(vis_frame)
            if save_out_video:
                videoWriter.release()

if __name__ == '__main__':
    video_path = '../my_data/1.avi'
    # pose_config = '../associative_embedding_hrnet_w32_coco_512x512.py'
    # pose_checkpoint = '../hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    pose_config = '../configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py'
    pose_checkpoint='../hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    out_img_root = 'results'
    video_kp(video_path,pose_config,pose_checkpoint,out_img_root)