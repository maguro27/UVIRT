import os
import glob
import random


def main():
    target = "test"
    random_bool = True
    repeat = 10

    img_side_paths = glob.glob("./datasets/MPV_supervised/{}/image/*".format(target))
    cloth_side_paths = glob.glob("./datasets/MPV_supervised/{}/cloth/*".format(target))
    for s in range(repeat):
        if random_bool:
            text_path = "./datasets/MPV_supervised/{0}_pairs_fid_{1}.txt".format(
                target, s
            )
            random.shuffle(img_side_paths)
            random.shuffle(cloth_side_paths)
        else:
            text_path = "./datasets/MPV_supervised/{}_pairs.txt".format(target)
            img_side_paths.sort()
            cloth_side_paths.sort()

        with open(text_path, mode="a") as f:
            for i in range(len(cloth_side_paths)):
                f.write(
                    os.path.basename(img_side_paths[i])
                    + " "
                    + os.path.basename(cloth_side_paths[i])
                    + "\n"
                )


if __name__ == "__main__":
    main()
