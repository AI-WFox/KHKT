import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import time

model_load_path = 'modelmodel7classvgg16.h5'
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 7)
model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
model.eval()

transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
class_labels = ["cardboard", "e-waste", "glass", "medical", "metal", "paper", "plastic"]

cap = cv2.VideoCapture(0)

last_detected_class = None
last_detection_time = None

def send_result_if_time_elapsed():
    global last_detection_time, last_detected_class

    if last_detected_class is not None:
        if time.time() - last_detection_time >= 5:
            return last_detected_class
    return None

def recognize_frames():
    global last_detection_time, last_detected_class

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transformations(frame_rgb)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            model.eval()
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        predicted_label = class_labels[predicted_class.item()]

        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Recognition", frame)

        current_time = time.time()
        if last_detected_class is None or predicted_label != last_detected_class:
            last_detection_time = current_time
            last_detected_class = predicted_label

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_frames()
