import cv2
import os

dataset = "HRF"

input_paths = [
    f"data/{dataset}/train/images",
    f"data/{dataset}/train/labels"
]
output_paths = [
    f"data/{dataset}/aug/images/flip",
    f"data/{dataset}/aug/labels/flip"
]

assert len(input_paths) == len(output_paths)

for input, output in zip(input_paths, output_paths):

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    for name in os.listdir(input):
        image = cv2.imread(os.path.join(input, name))

        cv2.imwrite(
            os.path.join(output, "h" + name),
            cv2.flip(image, 1)      # flip horizontally
        )
        cv2.imwrite(
            os.path.join(output, "v" + name),
            cv2.flip(image, 0)      # flip vertically
        )
        cv2.imwrite(
            os.path.join(output, "hv" + name),
            cv2.flip(image, -1)    # flip horizontally and vertically
        )


