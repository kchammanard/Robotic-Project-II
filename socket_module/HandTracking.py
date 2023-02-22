import mediapipe as mp
import cv2
import numpy as np
import time
import math

class HandTracking:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.model = self.mp_hands.Hands(
            min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2)

    def track(self, image):
        return self.model.process(image)

    def read_results(self, image, hands_results):
        self.frame_width, self.frame_height = int(
            image.shape[1]), int(image.shape[0])
        self.image = image
        self.hands_results = hands_results

    def get_distance(self, p1, p2):
        dx, dy, dz = (p2[i] - p1[i] for i in range(3))
        dxy = (dx ** 2 + dy ** 2) ** 0.5

        return dx, dy, dz, dxy

    # HAND

    def get_hand_coords(self, hand, landmark_index):
        return tuple(np.multiply(
            np.array(
                (hand.landmark[landmark_index].x, hand.landmark[landmark_index].y, hand.landmark[landmark_index].z)),
            [self.frame_width, self.frame_height, self.frame_width]).astype(int))

    def get_exact_hand_coords(self, hand, landmark_index):
        return tuple(np.multiply(
            np.array(
                (hand.landmark[landmark_index].x, hand.landmark[landmark_index].y, hand.landmark[landmark_index].z)),
            [self.frame_width, self.frame_height, self.frame_width]))

    def get_moy_coords(self, hand, landmark_index):
        return tuple((hand.landmark[landmark_index].x, hand.landmark[landmark_index].y))

    def get_hand_label(self, index, hand, results):
        classification = results.multi_handedness[index]
        label = classification.classification[0].label
        label = ("Right", "Left")[("Left", "Right").index(label)]
        score = classification.classification[0].score
        txt = "{} {}".format(label, round(score, 2))
        coords = self.get_hand_coords(hand, 0)[:2]

        return txt, coords

    def draw_finger_angles(self, image, hand, joint_list):
        for joint in joint_list:
            co1, co2, co3 = [self.get_hand_coords(
                hand, joint[i]) for i in range(3)]

            radxy = np.arctan2(co3[1] - co2[1], co3[0] - co2[0]) - \
                np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            anglexy = np.abs(radxy * 180 / np.pi)
            anglexy = min(anglexy, 360 - anglexy)

            cv2.putText(image, str(round(anglexy, 2)), co2[:2], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

        return image

    def get_hand_slope_angle(self, hand, index1, index2):
        co1, co2 = self.get_exact_hand_coords(
            hand, index1), self.get_exact_hand_coords(hand, index2)

        radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
        return radxy

    def get_hand_slope(self, hand, index1, index2):
        co1, co2 = self.get_exact_hand_coords(
            hand, index1), self.get_exact_hand_coords(hand, index2)
        slope = (co2[1] - co1[1]) / (co2[0] - co1[0])

        return slope

    def draw_cont_line(self, hand, image, start_point, mid_point, length=200, color=(0, 255, 0), thickness=2):
        co_mid = self.get_hand_coords(hand, mid_point)
        co_start = self.get_hand_coords(hand, start_point)
        slope = self.get_hand_slope(hand, start_point, mid_point)
        slope_angle = self.get_hand_slope_angle(hand, start_point, mid_point)

        if co_mid[0] >= co_start[0]:
            xlen = round(abs(math.cos(slope_angle) * length))
        else:
            xlen = -round(abs(math.cos(slope_angle) * length))

        if co_mid[1] >= co_start[1]:
            ylen = round(abs(math.sin(slope_angle) * length))
        else:
            ylen = -round(abs(math.sin(slope_angle) * length))

        cv2.line(image, co_mid[:2], (co_mid[0] + xlen,
                 co_mid[1] + ylen), color, thickness)

        return co_start, co_mid, slope

    def draw_hand(self):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):
            self.mp_drawing.draw_landmarks(self.image, hand, self.mp_hands.HAND_CONNECTIONS,
                                           self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                           self.mp_drawing_styles.get_default_hand_connections_style())

    def draw_box(self, image, box_name, xywh_tuple, is_pointed):
        x, y, w, h = xywh_tuple
        if is_pointed:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, box_name, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    def draw_boxes(self, box_list):
        for box in box_list:
            self.draw_box(self.image, *box)

    def draw_hand_label(self):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

            if self.get_hand_label(num, hand, self.hands_results):
                text, coord = self.get_hand_label(
                    num, hand, self.hands_results)
                cv2.putText(self.image, text, coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def point_to(self, box_list, finger_list):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

            for boxi in range(len(box_list)):
                box_name, xywh, is_pointed = box_list[boxi]
                bx, by, bw, bh = xywh
                for finger in finger_list:
                    co_start, co_mid, slope = self.draw_cont_line(
                        hand, self.image, *finger, color=(255, 0, 0))
                    # print(co_start, co_mid, slope)
                    finger_len = finger[2]

                    # y-intercept
                    c = co_mid[1] - slope * co_mid[0]

                    # get range of x and y
                    if co_start[0] >= co_mid[0]:
                        range_x = [0, co_mid[0]]
                    else:
                        range_x = [co_mid[0] + 1, self.frame_width]

                    if co_start[1] >= co_mid[1]:
                        range_y = [0, co_mid[1]]
                    else:
                        range_y = [co_mid[1] + 1, self.frame_height]

                    # if box in range x and y
                    if (range_x[0] <= bx <= range_x[1] or range_x[0] <= bx + bw <= range_x[1]) \
                            and (range_y[0] <= by <= range_y[1] or range_y[0] <= by + bh <= range_y[1]):
                        y_bx = slope * bx + c
                        y_bxw = slope * (bx + bw) + c

                        # if not line goes above or below box
                        if not ((y_bx < by and y_bxw < by) or (
                                y_bx > by + bh and y_bxw > by + bh)) and \
                                finger_len >= self.get_distance(co_mid, (bx + bw / 2, by + bh / 2, 0))[-1] - (
                                bw + bh) / 2:
                            box_list[boxi][-1] = True  # set is_pointed to True
        sol = []
        for b in box_list:
            if b[-1] is True:
                sol.append(b[0])

        if [b[0] for b in box_list if b[-1]]:
            cv2.putText(self.image, "Pointed at: " + ",".join([b[0] for b in box_list if b[-1]]),
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print("Pointed at {}".format([b[0] for b in box_list if b[-1]]))

        return sol