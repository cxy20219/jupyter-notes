import cv2
# 存放二进制数据
ob_list=[]
s="0b"

video = cv2.VideoCapture("BadApple.mp4")
ret,img = video.read()

import time
while ret:

    # 读取图片数据
    img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
    # 等比缩放图片
    img=cv2.resize(img,(128,64))
    # 二值化处理
    ret,img=cv2.threshold(img, 150, 1, cv2.THRESH_BINARY);

    img = img.reshape(1024,8)
    for i in img:
        for j in i:
            s+=str(j)
        ob_list.append(s)
        s="0b"  

    # 存放十六进制数据    
    ox_list=['0x'+'%02x' % int(i, 2) for i in ob_list]
    # j=0
    # for i in ox_list:
    #     j+=1
    #     print(i,end=",")
    #     if j%16==0:
    #         print("")
    # print("=="*20)
    # print(ox_list)
    # print("=="*20)
    # print(str(ox_list))
    ox_str = ",".join(ox_list)
    print(ox_str)
    with open("bit.txt","a") as f:
        f.write(ox_str)
        f.write("\n")
    # if cv2.waitKey(1009) & 0xFF == 27:
    #     break 
    time.sleep(2)
video.release()