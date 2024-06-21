import os


def save_test_txt(image_id_path, save_txt_path):
    image_ids = os.listdir(image_id_path)
    f = open(save_txt_path, 'w', encoding='utf-8')
    for image in image_ids:
        f.write(os.path.join(os.path.abspath(image_id_path), image))
        f.write('\n')
    f.close()
    return 0


if __name__ == '__main__':
    save_test_txt('images', 'test_txt.txt')