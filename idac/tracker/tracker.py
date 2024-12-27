import math as math
from idac.objectOfInterrest import ObjectOfInterrest
from idac.blobdet.blob_detector_factory import BlobDetectorFactory
import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
import time

#MAX_TRACKS = 500
#MAX_TRACKS = 700
MAX_TRACKS = 1000
#IMG_DIAG = 2203  # Diagonal of image  C920 camera with image size 1920x1080
IMG_DIAG = 4405  # Diagonal of image Brio camera with image size 3840x2160 

class Tracker:
    def __init__(self, conf):
        self.savedois = []
        self.detector = BlobDetectorFactory.get_blob_detector(conf['blobdetector']['type'], conf)
        self.conf = conf["tracker"]
        self.distance_cost_weight = self.conf["distance_cost_weight"]
        self.area_cost_weight = self.conf["area_cost_weight"]
        self.cosine_cost_weight = self.conf["cosine_cost_weight"]
        self.cost_thres = self.conf["cost_threshold"]
        self.maxage = self.conf["maxage"]
        self.maxtimesec = self.conf["maxtimesec"]
        self.appear = {}

    #Might be deprecated
    def init_frame(self, image):
        startid = 0
        image_new, count, ooi1, id = self.detector.findboxes(image, startid)
        for oi in ooi1:
            oi.id = startid
            startid = startid + 1
        return image_new, count, ooi1, id

    def calc_e_distance(self, x2, x1, y2, y1):
        return float(math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2)))

    def calc_area_ratio(self, w1, h1, w2, h2):
        area1 = w1 * h1
        area2 = w2 * h2
        max_area = max(area1, area2)
        min_area = min(area1, area2)
        return min_area / max_area
    
    def calc_cost(self, g, o):
        
        if g.features is None: # Calc cost using distance and area
            #cost = (self.calc_e_distance(g.x, o.x, g.y, o.y) / 2203) * self.distance_cost_weight + ( #C920
            cost = (self.calc_e_distance(g.x, o.x, g.y, o.y) / IMG_DIAG) * self.distance_cost_weight + ( 
                    1 - self.calc_area_ratio(g.w, g.h, o.w, o.h)) * self.area_cost_weight
        else: # Calc cost using distance, area and cosine similarity of features
            cosine = np.dot(g.features,o.features)/(norm(g.features)*norm(o.features))
            cost = (self.calc_e_distance(g.x, o.x, g.y, o.y) / IMG_DIAG) * self.distance_cost_weight + ( 
                    1 - self.calc_area_ratio(g.w, g.h, o.w, o.h)) * self.area_cost_weight + (
                    1 - cosine) * self.cosine_cost_weight
            #print(cost, cosine)
        return cost

    def calc_cost_matrix(self, oii1, oii2):
        cost_m = np.full((MAX_TRACKS, MAX_TRACKS), 8.1)
        row_index = 0
        # TODO Fix this plzzzz
        for g in oii1:
            col_index = 0
            for o in oii2:
                cost = self.calc_cost(g, o)                    
                cost_m[row_index][col_index] = cost
                col_index = col_index + 1
            row_index = row_index + 1
        return cost_m
    
    def calc_time_cost_matrix(self, oii1, oii2):
        cost_m = np.full((MAX_TRACKS, MAX_TRACKS), 8.1)
        row_index = 0
        # TODO Fix this plzzzz
        for g in oii1:
            col_index = 0
            for o in oii2:
                timediffsec =  o.timesec - g.timesec
                if timediffsec > self.maxtimesec:
                    cost = 2 # High cost since time more than maximum duration since last recording
                else:
                    cost = self.calc_cost(g, o)                    
                cost_m[row_index][col_index] = cost
                col_index = col_index + 1
            row_index = row_index + 1
        return cost_m
    
    def determine(self, obj):
        if self.appear[obj.id] >= self.maxage:
            self.appear.pop(obj.id, None)
            return False
        else:
            return True

    def check_age(self, oois):
        for obj in oois:
            if obj.id in self.appear:
                self.appear[obj.id] += 1
            else:
                self.appear[obj.id] = 1

        self.savedois[:] = [obj for obj in oois if self.determine(obj)]

    def track_frames(self, ooi1, image, id):
        ooi1.extend(self.savedois)
        goods = []
        toremoveold = []
        toremovenew = []
        image2_new, count2, ooi2, id, binary = self.detector.findboxes(image, id)
        time1 = time.time()
        cost = Tracker.calc_cost_matrix(self, ooi1, ooi2)
        row_ind, col_ind = linear_sum_assignment(cost)
        time2 = time.time()
        #KBE?? print('Cost took {:.3f} ms'.format((time2 - time1) * 1000.0))

        time1 = time.time()
        for i in range(len(row_ind)):
            if cost[row_ind[i]][col_ind[i]] < self.cost_thres:
                #print('Row id: ' + str(ooi1[row_ind[i]].id) + ' With: ' + str(col_ind[i]))
                obj = ObjectOfInterrest(ooi2[col_ind[i]].x, ooi2[col_ind[i]].y, ooi2[col_ind[i]].w,
                                        ooi2[col_ind[i]].h, ooi1[row_ind[i]].id)
                ooi1[row_ind[i]].x = ooi2[col_ind[i]].x
                ooi1[row_ind[i]].y = ooi2[col_ind[i]].y
                ooi1[row_ind[i]].w = ooi2[col_ind[i]].w
                ooi1[row_ind[i]].h = ooi2[col_ind[i]].h
                ooi1[row_ind[i]].updatecenterhist()
                obj = ooi1[row_ind[i]]
                goods.append(obj)
                toremovenew.append(col_ind[i])
                toremoveold.append(row_ind[i])
        for i in sorted(toremoveold, reverse=True):
            del ooi1[i]

        for i in sorted(toremovenew, reverse=True):
            del ooi2[i]

        for oi in ooi2:
            goods.append(ObjectOfInterrest(oi.x, oi.y, oi.w, oi.h, id))
            id = id + 1
        time2 = time.time()
        #KBE?? print('Tracker took {:.3f} ms'.format((time2 - time1) * 1000.0))
        # self.savedois = ooi1
        self.check_age(ooi1)

        return goods, id, binary

    def track_boxes(self, ooi1, ooi2, count2, id):
        ooi1.extend(self.savedois)
        goods = []
        toremoveold = []
        toremovenew = []
        #image2_new, count2, ooi2, id, binary = self.detector.findboxes(image, id)
        time1 = time.time()
        cost = Tracker.calc_time_cost_matrix(self, ooi1, ooi2)
        row_ind, col_ind = linear_sum_assignment(cost)
        time2 = time.time()
        #KBE?? print('Cost took {:.3f} ms'.format((time2 - time1) * 1000.0))

        time1 = time.time()
        for i in range(len(row_ind)):
            if cost[row_ind[i]][col_ind[i]] < self.cost_thres:
                #print('Row id: ' + str(ooi1[row_ind[i]].id) + ' With: ' + str(col_ind[i]))
                obj = ObjectOfInterrest(ooi2[col_ind[i]].x, ooi2[col_ind[i]].y, ooi2[col_ind[i]].w,
                                        ooi2[col_ind[i]].h, ooi1[row_ind[i]].id)
                ooi1[row_ind[i]].x = ooi2[col_ind[i]].x
                ooi1[row_ind[i]].y = ooi2[col_ind[i]].y
                ooi1[row_ind[i]].w = ooi2[col_ind[i]].w
                ooi1[row_ind[i]].h = ooi2[col_ind[i]].h
                ooi1[row_ind[i]].copy(ooi2[col_ind[i]])
                # ooi1[row_ind[i]].label = ooi2[col_ind[i]].label
                # ooi1[row_ind[i]].percent = ooi2[col_ind[i]].percent
                # ooi1[row_ind[i]].confidenceAvg = ooi2[col_ind[i]].confidenceAvg
                # ooi1[row_ind[i]].key = ooi2[col_ind[i]].key
                # ooi1[row_ind[i]].valid = ooi2[col_ind[i]].valid
                # ooi1[row_ind[i]].timesec = ooi2[col_ind[i]].timesec
                ooi1[row_ind[i]].updatecenterhist()
                obj = ooi1[row_ind[i]]
                goods.append(obj)
                toremovenew.append(col_ind[i])
                toremoveold.append(row_ind[i])
        for i in sorted(toremoveold, reverse=True):
            del ooi1[i]

        for i in sorted(toremovenew, reverse=True):
            del ooi2[i]

        for oi in ooi2:
            obj = ObjectOfInterrest(oi.x, oi.y, oi.w, oi.h, id)
            obj.copy(oi)
            # obj.label = oi.label
            # obj.percent = oi.percent
            # obj.confidenceAvg = oi.confidenceAvg
            # obj.key = oi.key
            # obj.valid = oi.valid
            # obj.timesec = oi.timesec
            goods.append(obj)
            id = id + 1
        time2 = time.time()
        #KBE?? print('Tracker took {:.3f} ms'.format((time2 - time1) * 1000.0))
        # self.savedois = ooi1
        self.check_age(ooi1)

        return goods, id #, binary