from model import ROMPv1
import cv2
import numpy as np
import os, sys
import os.path as osp
import torch
from torch import nn
import argparse
import copy

from post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from utils import (
    img_preprocess,
    create_OneEuroFilter,
    euclidean_distance,
    check_filter_state,
    time_cost,
    download_model,
    determine_device,
    ResultSaver,
    WebcamVideoStream,
    convert_cam_to_3d_trans,
    wait_func,
    collect_frame_path,
    progress_bar,
    get_tracked_ids,
    smooth_results,
    convert_tensor2numpy,
    save_video_results,
)

from vis_human import setup_renderer, rendering_romp_bev_results
from post_parser import CenterMap

import time


def romp_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="ROMP: Monocular, One-stage, Regression of Multiple 3D People"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="image",
        help="Inferece mode, including image, video, webcam",
    )
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Path to the input image / video"
    )
    parser.add_argument(
        "-o",
        "--save_path",
        type=str,
        default=osp.join(osp.expanduser("~"), "ROMP_results"),
        help="Path to save the results",
    )
    parser.add_argument(
        "--GPU",
        type=int,
        default=0,
        help="The gpu device number to run the inference on. If GPU=-1, then running in cpu mode",
    )
    parser.add_argument(
        "--onnx", action="store_true", help="Whether to use ONNX for acceleration."
    )

    parser.add_argument(
        "-t",
        "--temporal_optimize",
        action="store_true",
        help="Whether to use OneEuro filter to smooth the results",
    )
    parser.add_argument(
        "--center_thresh",
        type=float,
        default=0.25,
        help="The confidence threshold of positive detection in 2D human body center heatmap.",
    )
    parser.add_argument(
        "--show_largest",
        action="store_true",
        help="Whether to show the largest person only",
    )
    parser.add_argument(
        "-sc",
        "--smooth_coeff",
        type=float,
        default=3.0,
        help="The smoothness coeff of OneEuro filter, the smaller, the smoother.",
    )
    parser.add_argument(
        "--calc_smpl",
        action="store_false",
        help="Whether to calculate the smpl mesh from estimated SMPL parameters",
    )
    parser.add_argument(
        "--render_mesh",
        action="store_true",
        help="Whether to render the estimated 3D mesh mesh to image",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="sim3dr",
        help="Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Whether to show the rendered results"
    )
    parser.add_argument(
        "--show_items",
        type=str,
        default="mesh",
        help="The items to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf. splited with ,",
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Whether to save the video results"
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=24,
        help="The frame_rate of saved video results",
    )
    parser.add_argument(
        "--smpl_path",
        type=str,
        default=osp.join(osp.expanduser("~"), ".romp", "SMPL_NEUTRAL.pth"),
        help="The path of smpl model file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=osp.join(osp.expanduser("~"), ".romp", "ROMP.pkl"),
        help="The path of ROMP checkpoint",
    )
    parser.add_argument(
        "--model_onnx_path",
        type=str,
        default=osp.join(osp.expanduser("~"), ".romp", "ROMP.onnx"),
        help="The path of ROMP onnx checkpoint",
    )
    parser.add_argument(
        "--root_align",
        type=bool,
        default=False,
        help="Please set this config as True to use the ROMP checkpoints trained by yourself.",
    )
    parser.add_argument("--webcam_id", type=int, default=0, help="The Webcam ID.")
    args = parser.parse_args(input_args)

    if not torch.cuda.is_available():
        args.GPU = -1
        args.temporal_optimize = False
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True
    if not os.path.exists(args.smpl_path):
        if os.path.exists(
            args.smpl_path.replace("SMPL_NEUTRAL.pth", "smpl_packed_info.pth")
        ):
            args.smpl_path = args.smpl_path.replace(
                "SMPL_NEUTRAL.pth", "smpl_packed_info.pth"
            )
        print(
            "please prepare SMPL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation"
        )
    if not os.path.exists(args.model_path):
        romp_url = "https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl"
        download_model(romp_url, args.model_path, "ROMP")
    if not os.path.exists(args.model_onnx_path) and args.onnx:
        romp_onnx_url = (
            "https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.onnx"
        )
        download_model(romp_onnx_url, args.model_onnx_path, "ROMP")
    return args


default_settings = romp_settings(input_args=[])


class ROMP(nn.Module):
    def __init__(self, romp_settings):
        super(ROMP, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()

    def _build_model_(self):
        if not self.settings.onnx:
            model = ROMPv1().eval()
            model.load_state_dict(
                torch.load(self.settings.model_path, map_location=self.tdevice)
            )
            model = model.to(self.tdevice)
            self.model = nn.DataParallel(model)
        else:
            try:
                import onnxruntime
            except:
                print(
                    "To use onnx model, we need to install the onnxruntime python package. Please install it by youself if failed!"
                )
                if not torch.cuda.is_available():
                    os.system("pip install onnxruntime")
                else:
                    os.system("pip install onnxruntime-gpu")
                import onnxruntime
            print("creating onnx model")
            self.ort_session = onnxruntime.InferenceSession(
                self.settings.model_onnx_path,
                providers=[
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            print("created!")

    def _initilization_(self):
        self.centermap_parser = CenterMap(conf_thresh=self.settings.center_thresh)

        if self.settings.calc_smpl:
            self.smpl_parser = SMPL_parser(self.settings.smpl_path).to(self.tdevice)

        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_()

        if self.settings.render_mesh:
            self.visualize_items = self.settings.show_items.split(",")
            self.renderer = setup_renderer(name=self.settings.renderer)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        if self.settings.onnx:
            center_maps, params_maps = self.ort_session.run(
                None, {"image": input_image.numpy().astype(np.float32)}
            )
            center_maps, params_maps = torch.from_numpy(center_maps).to(
                self.tdevice
            ), torch.from_numpy(params_maps).to(self.tdevice)
        else:
            center_maps, params_maps = self.model(input_image.to(self.tdevice))
        params_maps[:, 0] = torch.pow(1.1, params_maps[:, 0])
        parsed_results = parsing_outputs(
            center_maps, params_maps, self.centermap_parser
        )
        return parsed_results, image_pad_info

    def _initialize_optimization_tools_(self):
        self.OE_filters = {}
        if not self.settings.show_largest:
            try:
                from norfair import Tracker
            except:
                print(
                    "To perform temporal optimization, installing norfair for tracking."
                )
                os.system("pip install norfair")
                from norfair import Tracker
            self.tracker = Tracker(
                distance_function=euclidean_distance, distance_threshold=200
            )  # 120
            self.tracker_initialized = False

    def temporal_optimization(self, outputs, signal_ID):
        check_filter_state(
            self.OE_filters,
            signal_ID,
            self.settings.show_largest,
            self.settings.smooth_coeff,
        )
        if self.settings.show_largest:
            max_id = torch.argmax(outputs["cam"][:, 0])
            outputs["smpl_thetas"], outputs["smpl_betas"], outputs["cam"] = (
                smooth_results(
                    self.OE_filters[signal_ID],
                    outputs["smpl_thetas"][max_id],
                    outputs["smpl_betas"][max_id],
                    outputs["cam"][max_id],
                )
            )
            outputs["smpl_thetas"], outputs["smpl_betas"], outputs["cam"] = (
                outputs["smpl_thetas"].unsqueeze(0),
                outputs["smpl_betas"].unsqueeze(0),
                outputs["cam"].unsqueeze(0),
            )
        else:
            pred_cams = outputs["cam"]
            from norfair import Detection

            detections = [
                Detection(points=cam[[2, 1]] * 512) for cam in pred_cams.cpu().numpy()
            ]
            if not self.tracker_initialized:
                for _ in range(8):
                    tracked_objects = self.tracker.update(detections=detections)
            tracked_objects = self.tracker.update(detections=detections)
            if len(tracked_objects) == 0:
                return outputs
            tracked_ids = get_tracked_ids(detections, tracked_objects)
            for ind, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(
                        self.settings.smooth_coeff
                    )

                (
                    outputs["smpl_thetas"][ind],
                    outputs["smpl_betas"][ind],
                    outputs["cam"][ind],
                ) = smooth_results(
                    self.OE_filters[signal_ID][tid],
                    outputs["smpl_thetas"][ind],
                    outputs["smpl_betas"][ind],
                    outputs["cam"][ind],
                )

            outputs["track_ids"] = np.array(tracked_ids).astype(np.int32)
        return outputs

    # * original
    # @time_cost('ROMP')
    # def forward(self, image, signal_ID=0, **kwargs):
    #     outputs, image_pad_info = self.single_image_forward(image)
    #     if outputs is None:
    #         return None
    #     if self.settings.temporal_optimize:
    #         outputs = self.temporal_optimization(outputs, signal_ID)
    #     outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])
    #     if self.settings.calc_smpl:
    #         outputs = self.smpl_parser(outputs, root_align=self.settings.root_align)
    #         outputs.update(body_mesh_projection2image(outputs['joints'], outputs['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info))
    #     if self.settings.render_mesh:
    #         rendering_cfgs = {'mesh_color':'identity', 'items': self.visualize_items, 'renderer': self.settings.renderer} # 'identity'
    #         outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
    #     if self.settings.show:
    #         cv2.imshow('rendered', outputs['rendered_image'])
    #         wait_func(self.settings.mode)
    #     return convert_tensor2numpy(outputs)

    # * modified
    @time_cost("ROMP")
    def forward(
        self,
        image,
        is_interpolated,
        pre_raw_outputs,
        pre_image_pad_info,
        signal_ID=0,
        **kwargs,
    ):
        # print(is_interpolated)
        if is_interpolated:
            outputs, image_pad_info = pre_raw_outputs, pre_image_pad_info
            raw_outputs = None
        else:
            outputs, image_pad_info = self.single_image_forward(image)
            raw_outputs = copy.deepcopy(outputs)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
        outputs["cam_trans"] = convert_cam_to_3d_trans(outputs["cam"])
        if self.settings.calc_smpl:
            outputs = self.smpl_parser(outputs, root_align=self.settings.root_align)
            outputs.update(
                body_mesh_projection2image(
                    outputs["joints"],
                    outputs["cam"],
                    vertices=outputs["verts"],
                    input2org_offsets=image_pad_info,
                )
            )
        if self.settings.render_mesh:
            rendering_cfgs = {
                "mesh_color": "identity",
                "items": self.visualize_items,
                "renderer": self.settings.renderer,
            }  # 'identity'
            outputs = rendering_romp_bev_results(
                self.renderer, outputs, image, rendering_cfgs
            )
        if self.settings.show:
            cv2.imshow("rendered", outputs["rendered_image"])
            wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs), raw_outputs, image_pad_info

    # interpolation
    # @time_cost("ROMP")

    def interpolate(
        self,
        pre_output,
        pre_img_pad,
        image,
        # **kwargs,
    ):
        outputs = copy.deepcopy(pre_output)
        images_pad_info = copy.deepcopy(pre_img_pad)

        # file = open("out_pre.txt", "w")
        # file.write(f"output:\n{pre_output}\nimage_pad:\n{pre_img_pad}")
        # file.close()

        # file = open("out_post.txt", "w")
        # file.write(f"output:\n{post_output}\nimage_pad:\n{post_img_pad}")
        # file.close()

        # file = open("out.txt", "w")
        # file.write(f"output:\n{outputs}\nimage_pad:\n{image_pad_info}")
        # file.close()

        if outputs is None:
            return None

        outputs["cam_trans"] = convert_cam_to_3d_trans(outputs["cam"])
        if self.settings.calc_smpl:
            outputs = self.smpl_parser(outputs, root_align=self.settings.root_align)
            outputs.update(
                body_mesh_projection2image(
                    outputs["joints"],
                    outputs["cam"],
                    vertices=outputs["verts"],
                    input2org_offsets=images_pad_info,
                )
            )
        if self.settings.render_mesh:
            rendering_cfgs = {
                "mesh_color": "identity",
                "items": self.visualize_items,
                "renderer": self.settings.renderer,
            }  # 'identity'
            outputs = rendering_romp_bev_results(
                self.renderer,
                outputs,
                image,
                rendering_cfgs,
            )
        if self.settings.show:
            cv2.imshow("rendered", outputs["rendered_image"])
            wait_func(self.settings.mode)

        return convert_tensor2numpy(outputs)


def main():
    args = romp_settings()
    romp = ROMP(args)
    if args.mode == "image":
        saver = ResultSaver(args.mode, args.save_path)
        image = cv2.imread(args.input)
        outputs = romp(image)
        saver(outputs, args.input)

    default = False

    if args.mode == "video":
        frame_paths, video_save_path = collect_frame_path(args.input, args.save_path)
        num_frames = len(frame_paths)
        saver = ResultSaver(args.mode, args.save_path)
        # default, make pose on every frame
        if default:
            start_time = time.time()
            for frame_path in progress_bar(frame_paths):
                # print(f"frame path: {frame_path}")
                image = cv2.imread(frame_path)
                outputs, raw_outputs, image_pad_info = romp(image, False, 0, 0)
                # if counter == 1:
                #     file = open("out_org.txt", "w")
                #     file.write(f"output:\n{raw_outputs}\nimage_pad:\n{image_pad_info}")
                #     file.close()
                saver(outputs, frame_path)
                # counter += 1

            runtime = time.time() - start_time
            print(f"runtime: {runtime:.4f} s")
            print(f"FPS: {(num_frames/runtime):.2f}")

        # interpolate
        else:
            is_interpolate = False
            pre_raw_outputs, pre_image_pad_info = 0, 0

            start_time_overall = time.time()
            for frame_path in frame_paths:
                print(frame_path)
                image = cv2.imread(frame_path)
                outputs, pre_raw_outputs, pre_image_pad_info = romp(
                    image,
                    is_interpolate,
                    pre_raw_outputs,
                    pre_image_pad_info,
                )
                saver(outputs, frame_path)
                is_interpolate = not is_interpolate
                # check if it is interpolated frame
                # if counter % 2 == 0:
                #     outputs, pre_raw_outputs, pre_image_pad_info = romp(image)
                #     saver(outputs, frame_path)
                # else:
                #     start_time = time.time()
                #     outputs = romp.interpolate(
                #         pre_raw_outputs, pre_image_pad_info, image
                #     )
                #     interpolate_runtime = time.time() - start_time
                #     print(
                #         f"Runtime: {(interpolate_runtime):.4f}, FPS: {(1/interpolate_runtime):.2f}"
                #     )
                #     saver(outputs, frame_path)
                # counter += 1
                # if counter % (interval - 1) != 0 and counter < limited:
                #     interpolate_frames.append(
                #         {
                #             "frame_path": copy.deepcopy(frame_path),
                #             "image": copy.deepcopy(image),
                #         }
                #     )
                #     # interpolate_frame_path, interpolate_image = copy.deepcopy(
                #     #     frame_path
                #     # ), copy.deepcopy(image)
                #     # continue
                # else:
                #     # if not, let it processed by model

                #     # if the frames needs to be interpolated
                #     if counter < limited:
                #         # information of first frame
                #         if counter == 0:
                #             pre_raw_outputs, pre_image_pad_info = copy.deepcopy(
                #                 raw_outputs
                #             ), copy.deepcopy(image_pad_info)
                #         # information of third frame
                #         else:
                #             post_raw_outputs, post_image_pad_info = copy.deepcopy(
                #                 raw_outputs
                #             ), copy.deepcopy(image_pad_info)

                #             # then interpolate
                #             # print(f"interpolate {interpolate_frame_path}")
                #             start_time = time.time()
                #             outputs = romp.interpolate(
                #                 pre_raw_outputs,
                #                 pre_image_pad_info,
                #                 post_raw_outputs,
                #                 post_image_pad_info,
                #                 interpolate_frames,
                #                 # do_change=(True if counter == 2 else False),
                #             )

                #             runtime = time.time() - start_time
                #             print(
                #                 f"runtime: {runtime:.4f}, FPS: {((interval - 2)/runtime):.2f}"
                #             )
                #             for output in outputs:
                #                 saver(output["outputs"], output["frame_path"])

                #             # if counter == 2:
                #             #     file = open("out_pre.txt", "w")
                #             #     file.write(
                #             #         f"output:\n{pre_raw_outputs}\nimage_pad:\n{pre_image_pad_info}"
                #             #     )
                #             #     file.close()

                #             #     file = open("out_post.txt", "w")
                #             #     file.write(
                #             #         f"output:\n{post_raw_outputs}\nimage_pad:\n{post_image_pad_info}"
                #             #     )
                #             #     file.close()

                #             #     file = open("out.txt", "w")
                #             #     file.write(
                #             #         f"output:\n{raw_outputs}\nimage_pad:\n{image_pad_info}"
                #             #     )
                #             #     file.close()

                #             # and make the post frame become the pre frame for the next interpolation
                #             pre_raw_outputs, pre_image_pad_info = copy.deepcopy(
                #                 post_raw_outputs
                #             ), copy.deepcopy(post_image_pad_info)

                #             outputs, interpolate_frames = [], []

            # interpolate frames
            # counter = 1
            # for (
            #     pre_raw_outputs,
            #     pre_image_pad_info,
            #     post_raw_outputs,
            #     post_image_pad_info,
            #     interpolate_frame_path,
            #     interpolate_image,
            # ) in interpolate_frames:
            #     outputs = romp.interpolate(
            #         pre_raw_outputs,
            #         pre_image_pad_info,
            #         post_raw_outputs,
            #         post_image_pad_info,
            #         interpolate_image,
            #     )
            #     saver(outputs, interpolate_frame_path)
            #     print(f"{counter}/{len(interpolate_frames)}: {interpolate_frame_path}")
            #     counter += 1

            runtime = time.time() - start_time_overall
            print(f"runtime: {runtime:.4f} s")
            print(f"FPS: {(num_frames/runtime):.2f}")
        # print(f"limited: {limited}")
        # print(f"num frames: {num_frames}")

        save_video_results(saver.frame_save_paths)
        if args.save_video:
            print(f"save video at: {video_save_path}")
            saver.save_video(video_save_path, frame_rate=args.frame_rate)

    if args.mode == "webcam":
        cap = WebcamVideoStream(args.webcam_id)
        cap.start()
        while True:
            frame = cap.read()
            outputs = romp(frame)
        cap.stop()


if __name__ == "__main__":
    print(f"run romp")
    main()
