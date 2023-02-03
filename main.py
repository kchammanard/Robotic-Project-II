import time
import cv2

IMG_PATH = "imglib"

pTime = 0
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if success:

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)

        # Saving image to a file
        cv2.imwrite(f"{IMG_PATH}/img.jpg", img)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
