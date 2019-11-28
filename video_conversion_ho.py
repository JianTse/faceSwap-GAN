import keras.backend as K
from networks.faceswap_gan_model import FaceswapGANModel
from converter.video_converter import VideoConverter
from detector.face_detector import MTCNNFaceDetector
import sys
def conversion(video_path_A,video_path_B):

    if len(sys.argv)<=2:
        output_video_path_A = "./Aout.mp4"
        output_video_path_B = "./Bout.mp4"
    elif len(sys.argv)==3:
        output_video_path_A = sys.argv[3]
    elif len(sys.argv)>=4:
        output_video_path_A = sys.argv[3]
        output_video_path_B = sys.argv[4]

    output_video_path_A = "./video/Aout.mp4"
    output_video_path_B = "./video/Bout.mp4"

    K.set_learning_phase(0)

    # Input/Output resolution
    RESOLUTION = 128 # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

    # Architecture configuration
    arch_config = {}
    arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
    arch_config['use_self_attn'] = True
    arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
    arch_config['model_capacity'] = "standard" # standard, lite

    model = FaceswapGANModel(**arch_config)

    model.load_weights(path="./models")

    mtcnn_weights_dir = "./mtcnn_weights/"

    fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
    vc = VideoConverter()

    vc.set_face_detector(fd)
    vc.set_gan_model(model)

    options = {
        # ===== Fixed =====
        "use_smoothed_bbox": True,
        "use_kalman_filter": True,
        "use_auto_downscaling": False,
        "bbox_moving_avg_coef": 0.65,
        "min_face_area": 35 * 35,
        "IMAGE_SHAPE": model.IMAGE_SHAPE,
        # ===== Tunable =====
        "kf_noise_coef": 3e-3,
        "use_color_correction": "hist_match",
        "detec_threshold": 0.7,
        "roi_coverage": 0.9,
        "enhance": 0.5,
        "output_type": 1,
        "direction": "BtoA",
    }

    input_fn = video_path_B
    output_fn = output_video_path_B
    duration = None

    vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=duration)


    #options
    options = {
        # ===== Fixed =====
        "use_smoothed_bbox": True,
        "use_kalman_filter": True,
        "use_auto_downscaling": False,
        "bbox_moving_avg_coef": 0.65,
        "min_face_area": 35 * 35,
        "IMAGE_SHAPE": model.IMAGE_SHAPE,
        # ===== Tunable =====
        "kf_noise_coef": 3e-3,
        "use_color_correction": "hist_match",
        "detec_threshold": 0.7,
        "roi_coverage": 0.9,
        "enhance": 0.5,
        "output_type": 1,
        "direction": "AtoB",
    }

    input_fn = video_path_A
    output_fn = output_video_path_A
    duration = None

    vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=duration)

# conversion(sys.argv[1],sys.argv[2])
conversion("./video/AA.avi","./video/BB.avi")
