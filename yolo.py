import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from PIL import Image
from lxml import etree
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
# from nets.yolo_ECV import YoloBody

from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression
from utils.dataloader import YoloDataset, yolo_dataset_collate, LoadImages

'''
训练自己的数据集必看注释！
'''


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'E:\Python-Projection\yolo\yolox-pytorch\logs\yolox-s-jiaolan.pth',
        "classes_path": 'model_data/voc_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        "phi": 's',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.2,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = YoloBody(self.num_classes, self.phi, self.input_shape)  # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))  # 加载参数
        # self.net = self.net.half()  # 半精度
        self.net = self.net.eval()  # 测试
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

        zz = torch.rand(1, 3, 640, 640)
        if self.cuda:
            zz = zz.cuda()
        self.net(zz)  # 启动GPU

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou, method=1)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        thickness = 2
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    # def detect_dir(self, annotation_path, bs, save_path, xml_path):
    #     test_dataset = LoadImages(annotation_path, self.input_shape)
    #     test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     with torch.no_grad():
    #         # t = 0
    #         for path, images, image_shape in test_loader:
    #             if self.cuda:
    #                 images = images.cuda()
    #             # s = time.time()
    #             outputs = self.net(images)
    #             outputs = decode_outputs(outputs, self.input_shape)
    #             results = non_max_suppression(outputs, self.num_classes, self.input_shape,
    #                                           image_shape, self.letterbox_image, conf_thres=self.confidence,
    #                                           nms_thres=self.nms_iou)  # 原图上的尺度，上左下右。
    #             for i in range(len(path)):
    #                 image = Image.open(path[i])
    #                 image_name = os.path.basename(path[i])
    #
    #                 # print(f'{image_name}: {len(results[i])} detections')
    #                 if results[i] is None:
    #                     image.save(save_path + '/' + image_name)
    #                     continue
    #                 top_label = np.array(results[i][:, 6], dtype='int32')
    #                 top_conf = results[i][:, 5]
    #                 top_boxes = results[i][:, :4]
    #                 # ---------------------------------------------------------#
    #                 #   设置字体与边框厚度
    #                 # ---------------------------------------------------------#
    #                 font = ImageFont.truetype(font='model_data/simhei.ttf',
    #                                           size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #                 # thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
    #                 thickness = 2
    #                 for j, c in list(enumerate(top_label)):
    #                     predicted_class = self.class_names[int(c)]
    #                     box = top_boxes[j]
    #                     score = top_conf[j]
    #
    #                     top, left, bottom, right = box
    #
    #                     top = max(0, np.floor(top).astype('int32'))
    #                     left = max(0, np.floor(left).astype('int32'))
    #                     bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    #                     right = min(image.size[0], np.floor(right).astype('int32'))
    #
    #                     label = '{} {:.2f}'.format(predicted_class, score)
    #
    #                     draw = ImageDraw.Draw(image)
    #                     label_size = draw.textsize(label, font)
    #                     label = label.encode('utf-8')
    #                     # print(label, top, left, bottom, right)
    #
    #                     if top - label_size[1] >= 0:
    #                         text_origin = np.array([left, top - label_size[1]])
    #                     else:
    #                         text_origin = np.array([left, top + 1])
    #
    #                     for t in range(thickness):
    #                         draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self.colors[c])
    #                     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
    #                     draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
    #                     del draw
    #
    #                 # 画真实框
    #                 xml = xml_path + '/' + image_name[:-4] + '.xml'
    #                 parser = etree.XMLParser(remove_blank_text=True)
    #                 xml = etree.parse(xml, parser)  # 解析 ElementTree
    #                 objs = xml.findall('object')
    #                 for i, obj in enumerate(objs):
    #                     bndbox = obj.find('bndbox')
    #                     left = float(bndbox.find('xmin').text)
    #                     top = float(bndbox.find('ymin').text)
    #                     right = float(bndbox.find('xmax').text)
    #                     bottom = float(bndbox.find('ymax').text)
    #                     draw = ImageDraw.Draw(image)
    #                     # label_size = draw.textsize('ship', font)
    #                     for i in range(1):
    #                         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(0, 255, 0))
    #                     del draw
    #                 image.save(save_path + image_name)
    #         #     t += time.time() - s
    #         # print(t / len(test_dataset), len(test_dataset) / t)

    def detect_dir(self, annotation_path, bs, save_path, xml_path):
        test_dataset = LoadImages(annotation_path, self.input_shape)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with torch.no_grad():
            # t = 0
            result_metrics = []
            for img_idx, images, image_shape in test_loader:  # 遍历所有batches
                if self.cuda:
                    images = images.cuda()
                # s = time.time()
                # 对一个批次进行前向传播、解码、非极大值抑制
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)  # 原图上的尺度，上左下右。
                for i in range(len(images)):
                    img = images[i].permute(1, 2, 0).cpu()
                    image = Image.fromarray(np.array(np.uint8(img * 255)))
                    image_name = img_idx[i]

                    # print(f'{image_name}: {len(results[i])} detections')
                    if results[i] is None:
                        image.save(save_path + '/' + str(image_name) + '.png')
                        continue
                    top_label = np.array(results[i][:, 6], dtype='int32')
                    top_conf = results[i][:, 5]
                    top_boxes = results[i][:, :4]
                    # ---------------------------------------------------------#
                    #   设置字体与边框厚度
                    # ---------------------------------------------------------#
                    font = ImageFont.truetype(font='model_data/simhei.ttf',
                                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                    # thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
                    thickness = 2
                    for j, c in list(enumerate(top_label)):
                        predicted_class = self.class_names[int(c)]
                        box = top_boxes[j]
                        score = top_conf[j]

                        top, left, bottom, right = box

                        top = max(0, np.floor(top).astype('int32'))
                        left = max(0, np.floor(left).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                        right = min(image.size[0], np.floor(right).astype('int32'))

                        label = '{} {:.2f}'.format(predicted_class, score)

                        draw = ImageDraw.Draw(image)
                        label_size = draw.textsize(label, font)
                        label = label.encode('utf-8')
                        # print(label, top, left, bottom, right)

                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])

                        for t in range(thickness):
                            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=self.colors[c])
                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                        del draw

                    image.save(save_path + '/' + str(image_name) + '.png')



    def shipin_detect_dir(self, annotation_path, bs, save_path):
        test_dataset = LoadImages(annotation_path, self.input_shape)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with torch.no_grad():
            # t = 0
            result_metrics = []
            for img_idx, images, image_shape in test_loader:  # 遍历所有batches
                if self.cuda:
                    images = images.cuda()
                # s = time.time()
                # 对一个批次进行前向传播、解码、非极大值抑制
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)  # 原图上的尺度，上左下右。
                for i in range(len(img_idx)):  # 遍历一个batch的所有图片
                    if results[i] is None:
                        continue

                    results[i][:, 4] = img_idx[i]  # 在每张图片的所有框加上图片的索引
                    top_boxes = results[i]
                    t1 = np.maximum(0, np.floor(top_boxes[:, 1]).astype('int32'))
                    t2 = np.minimum(image_shape[i][1], np.floor(top_boxes[:, 3]).astype('int32'))
                    f1 = image_shape[i][0] - np.minimum(image_shape[i][0], np.floor(top_boxes[:, 2]).astype('int32'))
                    f2 = image_shape[i][0] - np.maximum(0, np.floor(top_boxes[:, 0]).astype('int32'))
                    top_boxes[:, 0] = t1
                    top_boxes[:, 1] = t2
                    top_boxes[:, 2] = f1
                    top_boxes[:, 3] = f2

                    result_metrics.append(top_boxes)  # 放入一张图的框

            torch.save(result_metrics, 'results')


    def get_FPS_(self, annotation_path, bs):
        test_dataset = LoadImages(annotation_path, self.input_shape)
        num_test = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        t = 0
        f = 0
        d = 0
        n = 0

        with torch.no_grad():
            for path, images, image_shape in test_loader:
                if self.cuda:
                    images = images.cuda()

                start = time.time()

                f1 = time.time()
                outputs = self.net(images)
                f += time.time() - f1

                d1 = time.time()
                outputs = decode_outputs(outputs, self.input_shape)
                d += time.time() - d1

                n1 = time.time()
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)
                n += time.time() - n1

                t += time.time() - start
        return t / num_test, f / num_test, d / num_test, n / num_test

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            # outputs = self.net(images)
            # outputs = decode_outputs(outputs, self.input_shape)
            # # ---------------------------------------------------------#
            # #   将预测框进行堆叠，然后进行非极大抑制
            # # ---------------------------------------------------------#
            # results = non_max_suppression(outputs, self.num_classes, self.input_shape,
            #                               image_shape, self.letterbox_image, conf_thres=self.confidence,
            #                               nms_thres=self.nms_iou)

        t1 = time.time()
        f = 0
        d = 0
        n = 0
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                f1 = time.time()
                outputs = self.net(images)
                f += (time.time() - f1)

                d1 = time.time()
                outputs = decode_outputs(outputs, self.input_shape)
                d += (time.time() - d1)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#

                n1 = time.time()
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou, method=3)
                n += (time.time() - n1)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time, f / test_interval, d / test_interval, n / test_interval

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)

        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou, method=3)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 5] * results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
