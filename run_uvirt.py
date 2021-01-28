import os
import argparse
from tqdm import tqdm

import torch
from torch.backends import cudnn

from gdwct.run import Run
from data_loader_mpv import get_loader
from gdwct.utils.util import save_img, ges_Aonfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./configs/mpv.yaml",
        help="The path of a config file.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        help="implementation mode, i.e. train or test",
    )
    parser.add_argument(
        "--load_model", "-l", action="store_true", help="load pre-trained model or not"
    )
    parser.add_argument(
        "--start_iteration",
        "-s",
        type=int,
        default=0,
        help="start iteration number. please set it when you turn on the 'load_model' argument.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="mpv_results_mpv_192_256",
        help="a name of saved weight.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/mpv/",
        help="a name of saved weight.",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="When you generate random sample for computing FID, turn on.",
    )
    opts = parser.parse_args()

    config = ges_Aonfig(opts.config)
    config["MODE"] = opts.mode
    config["LOAD_MODEL"] = opts.load_model
    config["START"] = opts.start_iteration
    config["SAVE_NAME"] = opts.save_name
    config["MODEL_SAVE_PATH"] = opts.model_save_path

    # For fast training
    cudnn.benchmark = True

    run = Run(config)
    # Overwrite a instance variable
    run.data_loader = get_loader(
        config["DATA_PATH"],
        crop_size=config["CROP_SIZE"],
        resize=config["RESIZE"],
        batch_size=config["BATCH_SIZE"],
        dataset=config["DATASET"],
        mode=config["MODE"],
        num_workers=config["NUM_WORKERS"],
    )

    assert config["MODE"] in ["train", "test"]
    if config["MODE"] == "train":
        run.train()
    else:
        run.test(config, opts, run)


def test(config, opts, run):
    print("test start")
    run.test_ready()

    repeat_num = 10 if opts.fid else 1  # random test 10 times for FID
    with torch.no_grad():
        for s in range(repeat_num):
            if opts.fid:
                data_list = opts.mode + "_pairs_fid_" + str(s) + ".txt"
            else:
                data_list = None

            data_loader_test = get_loader(
                config["DATA_PATH"],
                crop_size=config["CROP_SIZE"],
                resize=config["RESIZE"],
                batch_size=config["BATCH_SIZE"],
                dataset=config["DATASET"],
                mode="test",
                num_workers=config["NUM_WORKERS"],
                data_list=data_list,
            )

            device = torch.device(
                "cuda:%d" % (int(config["GPU1"]))
                if torch.cuda.is_available()
                else "cpu"
            )
            for i, (x_A, x_B, im_name) in enumerate(tqdm(data_loader_test)):
                x_A = x_A.to(device)
                x_B = x_B.to(device)

                x_AB, x_BA, x_ABA, x_BAB = run.update_G(x_A, x_B, isTrain=False)

                if opts.fid:
                    save_img(
                        [x_BA],
                        self.config["SAVE_NAME"],
                        os.path.splitext(os.path.basename(im_name[0]))[0],
                        "test_results/" + opts.notice + "/fid_" + str(s),
                        opts.config,
                    )
                else:
                    if config["SAVE_IMG"]:
                        save_img(
                            [x_BA],
                            config["SAVE_NAME"],
                            os.path.splitext(os.path.basename(im_name[0]))[0],
                            "test_results",
                            opts.config,
                        )


if __name__ == "__main__":
    main()