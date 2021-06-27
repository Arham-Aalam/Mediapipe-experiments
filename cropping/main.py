import mediapipe as mp
import cv2
import numpy as np

def main():
    drawingModule = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands

    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        _, frameWidth, frameHeight = frame.shape[::-1]
        points_collector, cropping_enabled, initVal, is_holded = [], False, None, False
        cropped = None
        while ret:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    # for point in handsModule.HandLandmark:
                    #     print(point)
                    #     normalizedLandmark = handLandmarks.landmark[point]
                    #     pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
                    normalizedThumb = handLandmarks.landmark[handsModule.HandLandmark.THUMB_TIP]
                    pixelCoordinatesThumb = drawingModule._normalized_to_pixel_coordinates(normalizedThumb.x, normalizedThumb.y, frameWidth, frameHeight)
                    normalizedIndex = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP]
                    pixelCoordinatesIndex = drawingModule._normalized_to_pixel_coordinates(normalizedIndex.x, normalizedIndex.y, frameWidth, frameHeight)
                    
                    # print(pixelCoordinatesThumb, pixelCoordinatesIndex)
                    frame = cv2.circle(frame, pixelCoordinatesThumb, 5, (0,0,255), -1)
                    frame = cv2.circle(frame, pixelCoordinatesIndex, 5, (0,0,255), -1)
                    
                    
                    # print(abs(pixelCoordinatesThumb[0] - pixelCoordinatesIndex[0]) + abs(pixelCoordinatesIndex[1] - pixelCoordinatesThumb[1]))
                    if None in [pixelCoordinatesIndex, pixelCoordinatesThumb]:
                        continue
                    if abs(pixelCoordinatesThumb[0] - pixelCoordinatesIndex[0]) + abs(pixelCoordinatesIndex[1] - pixelCoordinatesThumb[1]) < 15:
                        if not cropping_enabled:
                            if initVal is None:
                                initVal = pixelCoordinatesThumb
                            elif len(points_collector) > 20 and abs(initVal[0] - pixelCoordinatesThumb[0]) + abs(initVal[1] - pixelCoordinatesThumb[1]) < 10:
                                print("[INFO] CROPPING Finished!")
                                initVal = None
                                cropping_enabled = True
                            points_collector.append([list(pixelCoordinatesThumb)])
                        else:
                            contour = np.array(points_collector)
                            dist = cv2.pointPolygonTest(contour, pixelCoordinatesThumb, True)
                            if dist >= 0.0:
                                # print("[INFO] Inside polygon")
                                if not is_holded:
                                    rect = cv2.boundingRect(contour)
                                    x,y,w,h = rect
                                    croped = frame[y:y+h, x:x+w].copy()
                                    contour = contour - contour.min(axis=0)
                                    mask = np.zeros(croped.shape[:2], np.uint8)
                                    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1, cv2.LINE_AA)
                                    dst = cv2.bitwise_and(croped, croped, mask=mask)
                                    bg = np.ones_like(croped, np.uint8)*255
                                    cv2.bitwise_not(bg,bg, mask=mask)
                                    cropped = bg + dst
                                    print("[INFO] Done!")
                                    is_holded = True
                                    last_loc = pixelCoordinatesThumb
                            last_loc = pixelCoordinatesThumb
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            if cropped is not None:
                try:
                    x, y = last_loc
                    frame[y: y+cropped.shape[0], x: x+cropped.shape[1], :, ] = cropped
                    # last_loc = pixelCoordinatesThumb
                except:
                    # indexing issue
                    pass
                    # print(points_collector)
                    # drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
            pxy = None
            for xy in points_collector:
                if pxy is None:
                    pxy = xy
                    continue
                frame = cv2.line(frame, pxy[0], xy[0], (0,255,0), 2)
                pxy = xy
                # print('------')
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == ord('c'):
                cropping_enabled = True
            if k == ord('x'):
                points_collector, cropping_enabled, initVal, is_holded = [], False, None, False
                cropped = None
            if k == ord('q'):
                break
            ret, frame = capture.read()
        
        cv2.destroyAllWindows()
        capture.release()

if __name__ == "__main__":
    main()