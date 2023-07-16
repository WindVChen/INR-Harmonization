import cv2
import shutil
from tqdm import tqdm
from pathlib import Path

max_size = 2048
input_dataset_path = r'.\iHarmony4\HAdobe5k'
output_path = f'{input_dataset_path}_resized{max_size}_{max_size}'

input_dataset_path = Path(input_dataset_path)
output_path = Path(output_path)

assert not output_path.exists()

output_path.mkdir()
for subfolder in ['composite_images', 'masks', 'real_images']:
    (output_path / subfolder).mkdir()

for annotation_path in input_dataset_path.glob('*.txt'):
    shutil.copy(annotation_path, output_path / annotation_path.name)

images_list = sorted(input_dataset_path.rglob('*.jpg'))
images_list.extend(sorted(input_dataset_path.rglob('*.png')))

for x in tqdm(images_list):
    image = cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
    new_path = output_path / x.relative_to(input_dataset_path)

    new_width = max_size
    new_height = max_size

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    if x.suffix == '.jpg':
        cv2.imwrite(str(new_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(str(new_path), image)