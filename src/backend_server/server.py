from pandas import read_excel
from urllib.request import urlretrieve
import pandas as pd
from math import isnan
import os
import numpy as np
import cv2
import uuid
from loguru import logger

import requests
import json


class POSM_SERVER:
    def __init__(self) -> None:
        logger.debug('Start')
        self.paths_to_dir = self.get_paths_to_dir()
        self.name_data = 'new_data_pos.xlsx'
        self.name_of_sheet = 'Где оценивать'
        path_to_data = os.path.join(self.paths_to_dir['data'], self.name_data)
        self.df = read_excel(path_to_data, self.name_of_sheet)
        self.post_segment = 'http://127.0.0.1:5000/predict'
        self.post_classify = 'http://127.0.0.1:5001/predict'
        self.report = self.df.copy()
        self.generated_data = {
            "megafon": {"front": [], "cash_register": [], "showcase": [], "interior": []},
            "yota" : {"front": [], "cash_register": [], "showcase": [], "interior": []},
            'total_sum': [],
            'res': [],
            'count_of_zones': []
        }
        self.zones = ['front', 'cash_register', 'showcase', 'interior']
        self.clients_ops = ['megafon', 'yota']
        self.competitors_ops = ['beeline', 'mts', 'tele2']
        self.zone_translater = {"front": "Фасад", "cash_register": "Касса", "showcase": "Витрины", "interior": "Интерьер"}
        logger.debug('Load data have ended')
    
    def get_paths_to_dir(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        paths_to_dir = {'data': os.path.join(BASE_DIR, 'data'),
                      'images': os.path.join(BASE_DIR, 'images'),
                      'reports': os.path.join(BASE_DIR, 'reports'),
                      'seg_images': os.path.join(BASE_DIR, 'seg_images'),
                      'cropped_images': os.path.join(BASE_DIR, 'cropped_images')}
        return paths_to_dir


    def segment_image(self, local_image_path):
        data = {'image_path': local_image_path}
        response = requests.post(self.post_segment, json=data)
        response_data = json.loads(response.text)
        return response_data
    
    def classify_image(self, local_image_path):
        data = {'image_path': local_image_path}
        response = requests.post(self.post_classify, json=data)
        response_data = json.loads(response.text)
        return response_data

    def get_image_by_url(self, url) -> str:
        save_as = str(os.path.join(self.paths_to_dir['images'], url.rsplit('/', 1)[-1] + '.jpg'))
        urlretrieve(url, save_as)
        return save_as

    def is_nan(self, value) -> bool:
        if not isinstance(value, float):
            return False
        if not isnan(value):
            return False
        return True

    def get_images_for_pos(self, pos: dict) -> dict:
        new_pos = {}
        for zone, url in pos.items():
                if not self.is_nan(url):
                    path = self.get_image_by_url(url)
                    new_pos[zone] = path
                else:
                    new_pos[zone] = None
        return new_pos
    
    def segment_pos(self, pos: dict) -> dict:
        new_pos = {}
        for zone, path in pos.items():
            if not path is None:
                new_pos[zone] = self.segment_image(path)
                if new_pos[zone] == []:
                    new_pos[zone] = None
            else:
                new_pos[zone] = None
        return new_pos


    def crop_seg_image(self, coordinates, image_file, save_as):
        image = cv2.imread(image_file)
        if image is None:
            logger.debug(f"Failed to load image: {image_file}")

        height, width = image.shape[:2]

        points = np.array(coordinates).reshape(-1, 2)
        # points[:, 0] *= width
        # points[:, 1] *= height
        points = points.astype(np.int32)

        # Create a mask from the polygon coordinates
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a single-channel mask
        cv2.fillPoly(mask, [points], 255)
        # Apply the mask to the image
        segmented_area = cv2.bitwise_and(image, image, mask=mask)

        # Crop the image to the bounding box of the polygon
        x, y, w, h = cv2.boundingRect(points)
        cropped_segmented_area = segmented_area[y:y+h, x:x+w]

        cv2.imwrite(save_as, cropped_segmented_area)


    def calc_area_for_zone(self, annotation, image_file):
        def calc_area(x_coords, y_coords):
            contour = np.array(list(zip(x_coords, y_coords)), dtype=np.float32).reshape((-1, 1, 2))
            area = cv2.contourArea(contour)
            return area
        class_areas = {}
        for result in annotation:
            class_name = result['name']
            confidence = result['confidence']
            x_coords = result['segments']['x']
            y_coords = result['segments']['y']
            pixel_area = calc_area(x_coords, y_coords)

            # классификация класса другие
            # формирование coordinates
            if class_name == 'Другая реклама':
                coordinates = []
                for x, y in zip(x_coords, y_coords):
                    coordinates.extend((x, y))
                # порезать фото
                save_as = os.path.join(self.paths_to_dir['cropped_images'], str(uuid.uuid4()) + ".jpg")
                self.crop_seg_image(coordinates, image_file, save_as)
                ans = self.classify_image(save_as)
                class_name = ans["class_name"]
                os.remove(save_as)

            if class_name in class_areas:
                class_areas[class_name].append(pixel_area)
            else:
                class_areas[class_name] = [pixel_area]
        
        all_op = {'Реклама Megafon', 'Реклама Yota', 'mts', 'tele2', 'beeline'}
        total_areas_pos = {class_name: sum(areas)  for class_name, areas in class_areas.items()}
        total_areas = sum([area for area in total_areas_pos.values()])
        for name_op in all_op.difference(total_areas_pos.keys()):
            total_areas_pos[name_op] = 0
        total_areas_pos_prec = {class_name: area / total_areas for class_name, area in total_areas_pos.items()}
        return total_areas_pos_prec
    
    def calc_area_for_pos(self, pos: dict, pos_images: dict) -> dict:
        areas = dict()
        for zone, ann in pos.items():
            if ann:
                areas[zone] = self.calc_area_for_zone(ann, pos_images[zone])
            else:
                areas[zone] = {}
        return areas

    def calcl_res_par(self):
        cnt = 0
        summa = 0
        for op in self.clients_ops:
            for zone in self.zones:
                val = self.generated_data[op][zone][len(self.generated_data[op][zone]) - 1]
                if not isnan(val):
                    cnt += 1
                    summa += val
        self.generated_data['total_sum'].append(summa)
        cnt = cnt // 2
        self.generated_data['count_of_zones'].append(cnt)
        if cnt == 0:
            self.generated_data['res'].append("Диспаритет")
        elif cnt == 1:
            if summa < 2:
                self.generated_data['res'].append("Диспаритет")
            elif summa < 4:
                self.generated_data['res'].append("Паритет")
            else:
                self.generated_data['res'].append("Приоритет")
        elif cnt == 2:
            if summa < 4:
                self.generated_data['res'].append("Диспаритет")
            elif summa < 7:
                self.generated_data['res'].append("Паритет")
            else:
                self.generated_data['res'].append("Приоритет")
        elif cnt == 3:
            if summa < 5:
                self.generated_data['res'].append("Диспаритет")
            elif summa < 10:
                self.generated_data['res'].append("Паритет")
            else:
                self.generated_data['res'].append("Приоритет")
        elif cnt == 4:
            if summa < 6:
                self.generated_data['res'].append("Диспаритет")
            elif summa < 13:
                self.generated_data['res'].append("Паритет")
            else:
                self.generated_data['res'].append("Приоритет")


    def count_max_comp(self, total_areas_zone):
        return max([total_areas_zone[op] for op in self.competitors_ops])

    def generate_data(self, total_areas_pos) -> dict:
        threshhold = 0.1
        translater = {'megafon': 'Реклама Megafon', 'yota': 'Реклама Yota'}
        for op in self.clients_ops:
            for zone, total_areas_zone in total_areas_pos.items():
                if total_areas_zone == {}:
                    self.generated_data[op][zone].append(np.nan)
                    continue
                
                max_comp_area = self.count_max_comp(total_areas_zone)
                client_area = total_areas_zone[translater[op]]

                if max_comp_area == 0:

                    if client_area > 0:
                        self.generated_data[op][zone].append(2)
                    else:
                        self.generated_data[op][zone].append(0)

                else:
                    if abs(client_area - max_comp_area) < threshhold:
                        self.generated_data[op][zone].append(1)
                    elif client_area > max_comp_area:
                        self.generated_data[op][zone].append(2)
                    else:
                        self.generated_data[op][zone].append(0)
        self.calcl_res_par()

    def generate_report(self):
        for zone in self.zone_translater:
            for op in self.clients_ops:
                self.report.insert(len(self.report.columns),
                                'Оценка_' + op + '_' + self.zone_translater[zone],
                                self.generated_data['megafon'][zone], False)
        self.report.insert(len(self.report.columns),
                           'Количество зон',
                           self.generated_data['count_of_zones'], False)
        self.report.insert(len(self.report.columns),
                                'Баллы',
                                self.generated_data['total_sum'], False)
        self.report.insert(len(self.report.columns),
                                'Оценка_прог',
                                self.generated_data['res'], False)
        self.report.to_excel(str(os.path.join(self.paths_to_dir['reports'], 'report.xlsx')))

    @logger.catch
    def run(self) -> None:
        logger.debug("Start generate report")
        for idx in self.df.index:
            pos = {}
            for zone, trans_zone in self.zone_translater.items():
                pos[zone] = self.df[trans_zone][idx]
            pos_images = self.get_images_for_pos(pos)
            segmented_pos = self.segment_pos(pos_images)
            total_areas_pos = self.calc_area_for_pos(segmented_pos, pos_images)
            self.generate_data(total_areas_pos)
        self.generate_report()
        logger.debug("report is ready")


if __name__ == "__main__":
    server = POSM_SERVER()
    server.run()