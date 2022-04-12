import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


@dataclass
class Timestamp:
    minute: int
    second: int

    def __eq__(self, o):
        return self.minute == o.minute and self.second == o.second

    def __gt__(self, other):
        if self.minute > other.minute:
            return True
        if self.minute < other.minute:
            return False
        if self.second > other.second:
            return True


def load_dataset_from_video(
    video_path: str,
    label_path: str,
    cut_timestamp: Timestamp,
    frames_skipped: int = 30,
    size: tuple[int, int] = (200, 200),
) -> tuple[list[np.array], list[str]]:
    label_timestamp = get_label_timestamp(label_path)
    ts, label = label_timestamp.pop(0)

    cap = cv2.VideoCapture(video_path)

    n_frame = 0
    images, labels = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = Timestamp(n_frame // (30 * 60), n_frame % (30 * 60) // 30)
        n_frame += 1
        if current_timestamp > cut_timestamp:
            break

        if n_frame % frames_skipped == 0:
            ts = Timestamp(ts.minute, ts.second)
            ts.second += 1
            if label_timestamp and current_timestamp == label_timestamp[0][0]:
                # print(current_timestamp, ts)
                ts, label = label_timestamp.pop(0)

            img = cv2.resize(frame, size)

            images.append(img.flatten())
            labels.append(label)
            # print(label)

            cv2.imshow("frame", frame)
            # cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    return np.array(images), np.array(labels)


def get_label_timestamp(label_path: str) -> list[tuple[Timestamp, str]]:
    with open(label_path, "r") as f:
        lines = f.readlines()

    label_timestamp = []
    for line in lines:
        line = line.split("-")
        line[1] = line[1].strip()

        mins, secs = line[0].split(":")
        timestamp = Timestamp(int(mins), int(secs))

        label = line[1]

        label_timestamp.append((timestamp, label))

    return label_timestamp


def predict_video(video_path: str, model: object, frames_skipped: int = 30) -> str:
    pass


def main(read_from_disk: bool = False, selected_model: int = 0):
    # obtencion de los datos de entrenamiento
    images, labels = load_dataset_from_video("dataset/lento_1.MOV", "dataset/lento_1.txt", Timestamp(0, 10), 30)
    print(images.shape)
    print(labels.shape)
    # obtener features de las imagenes (Opcional)
    #
    # Entrenamiento de 3 modelos clasificadores (KNN, MLP, Random Forest)
    if read_from_disk:
        # load the model from disk
        models = pickle.load(open("models.pkl", "rb"))
    else:
        models = [
            KNeighborsClassifier(n_neighbors=3),
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
            RandomForestClassifier(n_estimators=100),
        ]
        # entrennamiento de los modelos
        for model in models:
            model.fit(images, labels)
        # save models to disk
        pickle.dump(models, open("models.pkl", "wb"))

    # Mostrar metricas de los modelos
    for model in models:
        print(model.score(images, labels))

    # Prueba de modelo en video
    model = models[selected_model]
    exit()
    predict_video("", model)


if __name__ == "__main__":
    main()
