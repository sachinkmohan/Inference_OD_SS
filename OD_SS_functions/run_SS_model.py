
#import Inference_Only_SS
import Inference_OD_SS_copy

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image_resized2 = cv2.resize(frame, (480, 300))
        image_resized3 = cv2.resize(frame, (480, 320))
        
        a = Inference_OD_SS_copy.model_OS(image_resized2)
        b = Inference_OD_SS_copy.model_SS(image_resized3)

        cv2.imshow('Input Images', a)
        cv2.imshow('prediction mask',b)

        cv2.waitKey(0)
        if cv2.waitKey(0):
            break
        '''
        if a == False:
            print('Object Detection complete')
            break
        elif b == False:
            print('Semantic Segmentation complete')
            break
        '''

    else:
        cap.release()
        break

cap.release()
cv2.destroyAllWindows()
