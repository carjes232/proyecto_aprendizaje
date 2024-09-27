import os
import time
import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cpu"):  # Use CPU by default
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres):
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        return result_img

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in [".jpg", ".jpeg", ".webp", ".bmp", ".png"]:
                image_names.append(apath)
    return image_names

def measure_fps(predictor, image_name, num_runs=10):
    inference_times = []
    for i in range(num_runs):
        start_time = time.time()
        meta, res = predictor.inference(image_name)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Run {i+1}/{num_runs} - Inference Time: {inference_time:.4f} seconds")
        inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / num_runs
    fps = 1 / avg_inference_time
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds, FPS: {fps:.2f}")
    
    # Save inference times to a file
    with open('inference_times.txt', 'w') as f:
        for idx, t in enumerate(inference_times, 1):
            f.write(f"Inference {idx}: {t:.4f} seconds\n")
        f.write(f"Average Inference Time: {avg_inference_time:.4f} seconds, FPS: {fps:.2f}\n")
    print("Inference times saved to 'inference_times.txt'")

def main():
    # Static configurations
    model_path = "./nanodet-plus-m_320.pth"
    config_path = "./nanodet-plus-m_320.yaml"
    image_path = "../bus.jpg"
    save_results = True
    num_runs = 10

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cpu")  # Use CPU

    if os.path.isdir(image_path):
        files = get_image_list(image_path)
    else:
        files = [image_path]
    files.sort()

    for image_name in files:
        measure_fps(predictor, image_name, num_runs)
        meta, res = predictor.inference(image_name)
        result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
        if save_results:
            save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            mkdir(0, save_folder)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)
            print(f"Annotated image saved to {save_file_name}")

if __name__ == "__main__":
    main()
