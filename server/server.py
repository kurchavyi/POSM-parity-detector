from pandas import read_excel
from urllib.request import urlretrieve
import pandas as pd
from math import isnan
import os
import numpy as np
import cv2

import requests
import json

class POSM_SERVER:
    def __init__(self) -> None:
        print('Start')
        self.df = read_excel("server/data/new_data_pos.xlsx", 'Где оценивать')
        self.post_segment = 'http://127.0.0.1:5000/predict'
        self.report = self.df.copy()
        self.generated_data = {
            "megafon": {"front": [], "cash_register": [], "showcase": [], "interior": []},
            "yota" : {"front": [], "cash_register": [], "showcase": [], "interior": []},
            'total_sum': [],
            'res': [],
            'count_of_zones': []
        }
        self.zones = ['front', 'cash_register', 'showcase', 'interior']
        self.ops = ['megafon', 'yota']
        self.zone_translater = {"front": "Фасад", "cash_register": "Касса", "showcase": "Витрины", "interior": "Интерьер"}
        print('Load data have ended')
    
    def segment_image(self, local_image_path):
        url = 'http://127.0.0.1:5000/predict'
        data = {'image_path': local_image_path}
        response = requests.post(url, json=data)
        response_data = json.loads(response.text)
        return response_data

    
    def get_image_by_url(self, url) -> str:
        save_as = str(os.path.join('server', 'images', url.rsplit('/', 1)[-1] + '.jpg'))
        # urlretrieve(url, save_as) # раскомментировать, когда фото будет доступно
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

    def calc_area_for_zone(self, annotation):
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

            if class_name in class_areas:
                class_areas[class_name].append(pixel_area)
            else:
                class_areas[class_name] = [pixel_area]
        
        all_op = {'Реклама Megafon', 'Реклама Yota', 'Другая реклама'}
        total_areas_pos = {class_name: sum(areas)  for class_name, areas in class_areas.items()}
        total_areas = sum([area for area in total_areas_pos.values()])
        for name_op in all_op.difference(total_areas_pos.keys()):
            total_areas_pos[name_op] = 0
        total_areas_pos_prec = {class_name: area / total_areas for class_name, area in total_areas_pos.items()}
        return total_areas_pos_prec
    
    def calc_area_for_pos(self, pos: dict) -> dict:
        areas = dict()
        for zone, ann in pos.items():
            if ann:
                areas[zone] = self.calc_area_for_zone(ann)
            else:
                areas[zone] = {}
        return areas

    def calcl_res_par(self):
        cnt = 0
        summa = 0
        for op in self.ops:
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


    def generate_data(self, total_areas_pos) -> dict:
        threshhold = 5
        translater = {'megafon': 'Реклама Megafon', 'yota': 'Реклама Yota'}
        for op in self.ops:
            for zone, total_areas_zone in total_areas_pos.items():
                if total_areas_zone == {}:
                    self.generated_data[op][zone].append(np.nan)

                elif total_areas_zone['Другая реклама'] == 0:

                    if total_areas_zone[translater[op]] > 0:
                        self.generated_data[op][zone].append(2)
                    else:
                        self.generated_data[op][zone].append(0)

                else:
                    if abs(total_areas_zone[translater[op]] - total_areas_zone['Другая реклама']) < threshhold:
                        self.generated_data[op][zone].append(1)
                    elif total_areas_zone[translater[op]] > total_areas_zone['Другая реклама']:
                        self.generated_data[op][zone].append(2)
                    else:
                        self.generated_data[op][zone].append(0)
        self.calcl_res_par()

    def generate_report(self):
        for zone in self.zone_translater:
            for op in self.ops:
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
        self.report.to_excel('server/reports/report.xlsx')


    def run(self) -> None:
        print("Start generate report")
        for idx in self.df.index:
            pos = {}
            for zone, trans_zone in self.zone_translater.items():
                pos[zone] = self.df[trans_zone][idx]
            pos = self.get_images_for_pos(pos)
            segmented_pos = self.segment_pos(pos)
            total_areas_pos = self.calc_area_for_pos(segmented_pos)
            self.generate_data(total_areas_pos)
        self.generate_report()
        print("report is ready")
            # в сегментед имаджес должно появиться сегментированное изображение
            # crop image на image-object
            # посчитать площадь для каждого класса +
            # сделать предикт, добавить в отчет
            # удалить промежуточные данные (фото, сегмент фото)
            # вероятно есть смысл еще сделать




    # def start(self):
    #     while True:
    #         # read command

if __name__ == "__main__":
    server = POSM_SERVER()
    server.run()