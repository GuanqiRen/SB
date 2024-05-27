import cv2
import os

class PersonDetector:
    def __init__(self, model, conf_threshold=0.2):
        self.model = model
        self.conf_threshold = conf_threshold
        self.person_class_id = self.get_person_class_id()

    def get_person_class_id(self):
        yolo_classes = list(self.model.names.values())
        return yolo_classes.index("person")

    def detect_persons(self, image_path):
        img = cv2.imread(image_path)
        results = self.model.predict(img, conf=self.conf_threshold)
        return img, results

    def crop_and_save_persons(self, image_path, save_dir='cropped_images'):
        img, results = self.detect_persons(image_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image_counter = 0

        person_imgs = []

        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                if int(box.cls[0]) == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = img[y1:y2, x1:x2]
                    cropped_image_path = os.path.join(save_dir, f'person_{image_counter}.jpg')
                    cv2.imwrite(cropped_image_path, cropped_img)
                    image_counter += 1
                    persion_imgs.append(cropped_img)

        print(f"Cropped and saved {image_counter} person images.")

        return person_imgs

