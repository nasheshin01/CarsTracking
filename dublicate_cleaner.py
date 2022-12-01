import imagehash

from PIL import Image
import cv2

HASH_SIZE = 8
LIMIT_BITS = 10

class DublicateCleaner:

    def clean(self, images):
        hashes = []
        for i, image in enumerate(images):
            try:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image_hash = imagehash.phash(pil_image, hash_size=HASH_SIZE)
                hashes.append((image, image_hash))
            except:
                print("Error with", i)

        if len(hashes) <= 0:
            return []

        unique_hashes = [hashes.pop(0)]
        for i, image_hash_pair in enumerate(hashes):
            try:
                is_hash_unique = True
                for j, unique_hash_pair in enumerate(unique_hashes):
                    unique_hash = unique_hash_pair[1]
                    current_hash = image_hash_pair[1]

                    if unique_hash - current_hash < LIMIT_BITS:
                        is_hash_unique = False
                        break

                if is_hash_unique:
                    unique_hashes.append(image_hash_pair)
            except:
                print("Error with grouping hash")

        cleaned_images = [unique_hash_pair[0] for unique_hash_pair in unique_hashes]
        return cleaned_images
