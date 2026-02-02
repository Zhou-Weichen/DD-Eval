import os
from datasets import load_dataset

def save_tiny_imagenet_by_label(
    output_root,
    split='validation'
):
    dataset = load_dataset(
        'slegroux/tiny-imagenet-200-clean',
        split=split
    )

    os.makedirs(output_root, exist_ok=True)

    class_counters = {}

    for idx, sample in enumerate(dataset):
        img = sample['image']   
        label = sample['label'] 

        class_dir = os.path.join(output_root, f"{label:03d}")  
        os.makedirs(class_dir, exist_ok=True)

        cnt = class_counters.get(label, 0)
        filename = f"{cnt:06d}.JPEG"
        class_counters[label] = cnt + 1

        save_path = os.path.join(class_dir, filename)
        img.save(save_path, format='JPEG')

        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(dataset)}")

    print("Done.")

if __name__ == '__main__':
    save_tiny_imagenet_by_label(
        output_root='dataset/tiny/train',
        split='train'
    )

    save_tiny_imagenet_by_label(
        output_root='dataset/tiny/val/images',
        split='validation'
    )