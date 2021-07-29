from openvino.inference_engine import IECore
import cv2
import numpy as np
import matplotlib.pyplot as plt

ie = IECore()
net = ie.read_network(
    model = 'face-detection-0200.xml',
    weights = 'face-detection-0200.bin'
)

model = ie.load_network(net, "CPU")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    resized = cv2.resize(frame, (256,256))
    transposed = np.transpose(resized, (2,0,1))
    input_image = np.array([transposed], dtype = np.float16)
    
    result = model.infer({'image':input_image})
    for i in result['detection_out'][0, 0]:
        tinggi, lebar, channel = frame.shape
        if i[2] > 0.5:
            cv2.rectangle(
                frame,
                (int(lebar*i[3]), int(tinggi*i[4])),
                (int(lebar*i[5]), int(tinggi*i[6])),
                (255, 255, 255),5
            )
    
    #if cv2.waitKey(1) & 0xFF == ord(' '):
    #    break
        
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

        
cam.release()
cv2.destroyAllWindows()