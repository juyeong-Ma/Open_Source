#Install scipy di CMD
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#Import Numpy
#Install Numpy di CMD
import numpy as np

#Install Imutills di CMD
import imutils

#Import OpenCV
import cv2

#Inisialisasi variabel midpoint
#Menentukan titik tengah dari objek yang akan diukur
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


#Mengaktifkan Kamera untuk Menampilkan Video Secara Realtime
cap = cv2.VideoCapture(0)


#Membuat Kondisi
#Apabila Kamera Aktif dan Video telah dimulai, Maka Jalankan Program di Bawah Ini
while (cap.read()):
        ref,frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        orig = frame[:1080,0:1920]
       
        #Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

        result_img = closing.copy()
        contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hitung_objek = 0

        
        #Mengkonversi Nilai Pembacaan Pixel ke Dalam Satuan CM
        pixelsPerMetric = None

        #Membuat Kondisi Perulangan
        #Inisialisasi Variabel cnt = counturs
        for cnt in contours:

            #Pembacaan Area Objek yang di Ukur
            area = cv2.contourArea(cnt)

            #Jika Area Kurang dari 1000 dan Lebih dari 12000  Pixel
            #Maka Lakukan Pengukuran
            if area < 1000 or area > 120000:
                continue

            #Menghitung kotak pembatas dari contours Objek
            orig = frame.copy()
            box = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

            
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            #Menggambar titik tengah pada objek
            cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

            #Menggambar garis pada titik tengah
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

            #Menghitung jarak Euclidean antara titik tengah
            lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            #Jika piksel pixelsPerMetric belum diinisialisasi, maka
            #Hitung sebagai rasio piksel terhadap metrik yang disediakan
            #Dalam hal ini CM
            if pixelsPerMetric is None:
                pixelsPerMetric = lebar_pixel
                pixelsPerMetric = panjang_pixel
            lebar = lebar_pixel
            panjang = panjang_pixel

            #Menggambarkan ukuran benda pada gambar
            cv2.putText(orig, "L: {:.1f}CM".format(lebar_pixel/25.5),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            cv2.putText(orig, "P: {:.1f}CM".format(panjang_pixel/25.5),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            #cv2.putText(orig,str(area),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
            hitung_objek+=1

        #Menampilkan Jumlah Objek yang terdeteksi
        cv2.putText(orig, "Jumlah: {}".format(hitung_objek),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
        cv2.imshow('Kamera',orig)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
