import os
import shutil

# root = r"G:\Datasets\Images Harmonization\LR_real_composite_images_99_DIH\all"
#
# all_path = os.listdir(os.path.join(root, "image"))
#
# with open(os.path.join(root, "dataset.txt"), mode='w') as f:
#     for im in all_path:
#         f.write(os.path.join('image', im) + "\n")
#
# print("Done!")


# Re-order dataset
with open(r"G:\Datasets\Images Harmonization\iHarmony4\IHD_test.txt", mode="r") as f:
    names = f.readlines()

for id in range(len(names)):
    names[id] = names[id].strip().split("/")[-1].split(".")[0]

root = r"G:\ComputerPrograms\Image_Harmonization\Supervised_Harmonization\logs\HINet_2048×2048_HAdobe5k\figs\-1"
os.makedirs(os.path.join(root, "reorder"), exist_ok=True)
allFiles = os.listdir(root)
for id, file in enumerate(allFiles):
    if "pred_harmonized_image" in file:
        name = names[int(file.split("_")[0])] + "_" + file
        shutil.copy(os.path.join(root, file), os.path.join(root, "reorder", name))
        shutil.copy(os.path.join(root, file).replace("pred_harmonized_image", "mask"), os.path.join(root, "reorder", name).replace("pred_harmonized_image", "mask"))
        shutil.copy(os.path.join(root, file).replace("pred_harmonized_image", "real"), os.path.join(root, "reorder", name).replace("pred_harmonized_image", "real"))
        shutil.copy(os.path.join(root, file).replace("pred_harmonized_image", "composite"), os.path.join(root, "reorder", name).replace("pred_harmonized_image", "composite"))

print("Done!")
