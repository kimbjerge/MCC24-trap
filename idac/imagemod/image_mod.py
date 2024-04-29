import cv2

class Imagemod:
    def __init__(self):
        print('image_mod ready')

    def drawoois(self, image, ois):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for oi in ois:
            
            remainer = oi.id % 3
            color = (255, 255, 0) # Yelow (default)
            if remainer == 1:
                color = (255, 165, 0) # Orange
            if remainer == 2:
                color = (150, 75, 0) # Brown
            
            if oi.order == "Lepidoptera Macros":
                color = (0, 255, 0) # Green
            if oi.order == "Lepidoptera Micros":
                color = (0, 0, 255) # Blue
                
            if oi.valid == False:
                color = (255, 0, 0) # Red
                               
            cv2.rectangle(image, (oi.x, oi.y), (oi.x + oi.w, oi.y + oi.h), color, 2)
            colorText = (0, 0, 0) # Black
            if oi.y < 30:
                cv2.putText(image, 'id: ' + str(oi.id) + ' ' + oi.label, (oi.x, oi.y + 30 + oi.h), font, 1,
                            colorText, 2, cv2.LINE_AA)
                cv2.putText(image, '%.2f%%' % oi.percent, (oi.x, oi.y+60+oi.h), font, 1, colorText, 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(image, 'id: ' + str(oi.id) + ' ' + oi.label, (oi.x, oi.y-30), font, 1, colorText, 2, cv2.LINE_AA)
                cv2.putText(image, '%.2f%%' % oi.percent, (oi.x, oi.y), font, 1, colorText, 2, cv2.LINE_AA)

            if len(oi.centerhist) > 3:
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 1], oi.centerhist[len(oi.centerhist) - 2],
                         color, 4)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 2], oi.centerhist[len(oi.centerhist) - 3],
                         color, 4)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 3], oi.centerhist[len(oi.centerhist) - 4],
                         color, 4)

            elif len(oi.centerhist) > 2:
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 1], oi.centerhist[len(oi.centerhist) - 2],
                         color, 4)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 2], oi.centerhist[len(oi.centerhist) - 3],
                         color, 4)
            elif len(oi.centerhist) > 1:
                cv2.line(image, oi.centerhist[len(oi.centerhist)-1], oi.centerhist[len(oi.centerhist)-2], color, 4)
        return image
