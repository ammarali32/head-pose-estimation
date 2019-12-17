import torch
from imutils import face_utils
import imutils
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
import torch.backends.cudnn as cudnn
import time
import xlwt 
from xlwt import Workbook
import onnx


STD_SIZE = 120
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]


focal_length= 640
focal_length1= 480
center = (320,240)
ok=-1
cam_matrix = np.array(
                         [[focal_length1, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],#left eyebow left corner
                         [1.330353, 7.122144, 6.903745],#left eyebow rightcorner
                         [-1.330353, 7.122144, 6.903745],#right eyebow leftcorner
                         [-6.825897, 6.760612, 4.402142],#right eyebow rightcorner
                         [5.311432, 5.485328, 3.987654],#left eye left corner
                         [1.789930, 5.393625, 4.413414],#left eye right corner
                         [-1.789930, 5.393625, 4.413414],#right eye left corner 
                         [-5.311432, 5.485328, 3.987654],#right eye right corner
                         [2.005628, 1.409845, 6.165652],# left tip nose
                         [-2.005628, 1.409845, 6.165652],# right tip nose
                         [2.774015, -2.080775, 5.048531],#left mouth corner
                         [-2.774015, -2.080775, 5.048531],#right mouth corner
                         [0.000000, -3.116408, 6.097667],# bottom lip
                         [0.000000, -7.415691, 4.070434]# chin
                        
                            ])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):
    
    
    image_pts = np.float32([
        [shape[0][17],shape[1][17]],
        [shape[0][21],shape[1][21]],
        [shape[0][22],shape[1][22]],
        [shape[0][26],shape[1][26]],
        [shape[0][36],shape[1][36]],
        [shape[0][39],shape[1][39]],
        [shape[0][42],shape[1][42]],
        [shape[0][45],shape[1][45]],
        [shape[0][31],shape[1][31]],
        [shape[0][35],shape[1][35]],
        [shape[0][48],shape[1][48]],
        [shape[0][54],shape[1][54]],
        [shape[0][57],shape[1][57]],
        [shape[0][8],shape[1][8]]])
    
    
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    id1=1
    ratio=1
    #wb = Workbook()
    #sheet1 = wb.add_sheet('Sheet 1')
    
    #sheet1.write(0,0,"distance1")
    #sheet1.write(0,1,"distance2")
    #sheet1.write(0,2,"angle")
    
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location='cpu')['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)??
    
    net =cv2.dnn.readNet("tiny-yolo-azface-fddb_82000.weights","tiny-yolo-azface-fddb.cfg")
    classes=[]
    model_dict = model.state_dict()
        
    for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    cudnn.benchmark = True
    model = model.cuda()    
    model.eval()
    
 
    with open("face.names","r")as f:
        
        classes = [line.strip()for line in f.readlines()]
        layers_names = net.getLayerNames()
        outputlayers= [layers_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
        
        font = cv2.FONT_HERSHEY_PLAIN
        cap = cv2.VideoCapture(0)
        
        
       
        
        
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
        

        
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_PLAIN
        
        time_now=time.time()
        frame_id=0
        while True:
            
            img_fp='11.jpg'
            _, img_ori = cap.read()
            height , width , channels = img_ori.shape
            frame_id+=1
        
            blob = cv2.dnn.blobFromImage(img_ori, 0.00392, (480,480),(0,0,0),True,crop= False)
           
            net.setInput(blob)
            outs = net.forward(outputlayers)
            print("")
            class_ids=[]
            confidences=[]
            boxes=[]
            
            shape=[]
            gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
            lex1=-1
            ley1=-1
            rex1=-1
            rey1=-1
            ley2=-1
            rey2=-1
            rects=[]
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence> 0.2:
                        center_x= int(detection[0] *width)
                        center_y=int(detection[1]* height)
                        w= int(detection[2] *width)
                        h= int(detection[3] * height)
                        x= int(center_x- w /2)
                        y= int(center_y -h /2)
                        rects.append(dlib.rectangle(int(x),int(y),x+w,y+h))
                        
                        
           

            space =20
            for rect in rects:
            
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
                
                img = crop_img(img_ori, roi_box)
               
                img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img).unsqueeze(0)
                with torch.no_grad():
                    input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                    

                
                
                pts68 = predict_68pts(param, roi_box)
                

           
                
            
                reprojectdst, euler_angle = get_head_pose(pts68)
                cv2.putText(img_ori, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (space, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 0, 0), thickness=2)
                cv2.putText(img_ori, "Y: " + "{:7.2f}".format(euler_angle[1,0]), (space, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 0, 0), thickness=2)
                cv2.putText(img_ori, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (space, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 0, 0), thickness=2)
                space+=150

           
                for x in range(0, 67):
                        cv2.circle(img_ori,(pts68[0][x],pts68[1][x]) , 1, (0, 0, 255), -1)


                if euler_angle[1,0] >-5 and euler_angle[1,0] <5:
                    p1=np.array((pts68[0][36],pts68[1][36],0))
                    p2=np.array((pts68[0][45],pts68[1][45],0))
                    ratio=11.0/np.linalg.norm(p1-p2)
                    ratio=np.abs(ratio)
                a=np.array((pts68[0][33],pts68[1][33],0))
                b= np.array((pts68[0][14],pts68[1][14],0))
                c= np.array((pts68[0][2],pts68[1][2],0))
            #if euler_angle[1,0]>0:
             #       sheet1.write(id1,0,np.linalg.norm(a-b))
              #      sheet1.write(id1,2,euler_angle[1,0])
               #     id1+=1
            #elif euler_angle[1,0]<0:
             #       sheet1.write(id1,1,np.linalg.norm(a-c))
              #      sheet1.write(id1,2,euler_angle[1,0])
               #     id1+=1

            elapsed_time = time.time()-time_now
            fps =frame_id / elapsed_time
            cv2.putText(img_ori,"FPS: "+str(fps),(10,100),font,3,(0,0,0),1)
            cv2.imshow("Image",img_ori)
            key =cv2.waitKey(1)
            if key == 27:
              break
        cap.release()
        cv2.destroyAllWindows()
        #wb.save('test.xls')


if __name__ == '__main__':   
    main()

