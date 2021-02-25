import sys
import logging
import os
import cv2
import csv
import numpy as np
from utils import write_image, key_action, init_cam
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2

model = keras.models.load_model('models/mdl_wts.hdf5')


def predict_frame(frame):
    frame = mobilenet_v2.preprocess_input(frame)
    frame = frame.reshape(1,224,224,3)
    result = model.predict(frame)
        
    return result


if __name__ == "__main__":
  
    #Read label mapping
    reader = csv.reader(open('label_mapping.csv', 'r'))
    labels_dict = {}
    for row in reader:
        k, v = row
        labels_dict[int(k)] = v

    # folder to write images to
    #out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)

    key = None

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 


    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = cap.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
            # get key event
            key = key_action()

            # draw a [224x224] rectangle into the middle of the frame
            cv2.rectangle(frame,(int(frame.shape[1]/2)-112,0+8),(int(frame.shape[1]/2)+112,224+8),(0,255,0),2)     

            image = frame[0+8: 224+8, (int(frame.shape[1]/2)-112): (int(frame.shape[1]/2)-112+224), :]
            results = predict_frame(image)
            
            sorted_index_array = np.argsort(results[0]) 
            sorted_array = results[0][sorted_index_array] 
            predictions = [labels_dict[k].split('.')[1].strip() for k in sorted_index_array[-3 :]]

            first_pair = f'{predictions[-1]} {round(sorted_array[-1]*100,1)}%'
            second_pair = f'{predictions[-2]} {round(sorted_array[-2]*100,1)}%'
            third_pair = f'{predictions[-3]} {round(sorted_array[-3]*100,1)}%'

            cv2.putText(frame, 'Hold the leaf into the green area', (130, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 250, 0), thickness=1)
            cv2.putText(frame, first_pair, (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 250, 0), thickness=1)
            cv2.putText(frame, second_pair, (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 250),thickness=1)
            cv2.putText(frame, third_pair, (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 250),thickness=1)

            #disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xff == ord('q'): 
                break            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        cap.release()
        cv2.destroyAllWindows()
