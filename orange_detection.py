import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pickle
from collections import defaultdict


def get_label_data(video_path: str) -> tuple[np.array, np.array]:
    df = pd.read_csv(f"{video_path.split('.')[0]}_labels.csv")
    return df.values[:, 1:], df.values[:, 0]


def get_image_features_from_video(video_path: str, frames: np.array, size: int = 256) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    image_features = pd.DataFrame(
        columns=[
            "area",
            "perimeter",
            "ratio",
            "h_max",
            "s_max",
            "v_max",
            "h_min",
            "s_min",
            "v_min",
            "h_mean",
            "s_mean",
            "v_mean",
        ]
    )
    for specific_frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, specific_frame)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        # cv2.imshow("Frame", frame)
        image_features = pd.concat([image_features, get_image_features(frame)], ignore_index=True)
    return image_features


def get_image_features(image: np.array) -> pd.DataFrame:
    # obtener features de la imagen
    src_image = image
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    bin_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(bin_image, kernel, iterations=1)
    image = cv2.dilate(eroded_image, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contours = [contour for contour in contours if cv2.contourArea(contour) > 60]
    centroids = []
    for contour in contours:
        moments = cv2.moments(contour)
        centroids.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
    closest_centroid_index = get_centroid_closest_to_center(centroids, image.shape[1] // 2)

    contour = contours[closest_centroid_index]
    c_area = cv2.contourArea(contour)
    c_perimeter = cv2.arcLength(contour, True)
    c_ratio = c_area / c_perimeter

    x, y, w, h = cv2.boundingRect(contour)
    hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=image)
    # change the 0 to 255
    inv_masked_image = masked_image.copy()
    np.place(inv_masked_image, masked_image < 1, [255])
    c_max_h = np.max(masked_image[y : y + h, x : x + w, 0])
    c_max_s = np.max(masked_image[y : y + h, x : x + w, 1])
    c_max_v = np.max(masked_image[y : y + h, x : x + w, 2])
    c_min_h = np.min(inv_masked_image[y : y + h, x : x + w, 0])
    c_min_s = np.min(inv_masked_image[y : y + h, x : x + w, 1])
    c_min_v = np.min(inv_masked_image[y : y + h, x : x + w, 2])
    c_mean = cv2.mean(hsv_image[y : y + h, x : x + w], bin_image[y : y + h, x : x + w])
    # cv2.imshow("src slice", src_image[y : y + h, x : x + w])
    # cv2.imshow("gray slice", gray_image[y : y + h, x : x + w])
    # cv2.imshow("bin slice", bin_image[y : y + h, x : x + w])
    # cv2.imshow("masked slice", inv_masked_image[y : y + h, x : x + w])
    # cv2.waitKey(0)
    # print(c_mean)
    return pd.DataFrame(
        [
            [
                c_area,
                c_perimeter,
                c_ratio,
                c_max_h,
                c_max_s,
                c_max_v,
                c_min_h,
                c_min_s,
                c_min_v,
                c_mean[0],
                c_mean[1],
                c_mean[2],
            ],
        ],
        columns=[
            "area",
            "perimeter",
            "ratio",
            "h_max",
            "s_max",
            "v_max",
            "h_min",
            "s_min",
            "v_min",
            "h_mean",
            "s_mean",
            "v_mean",
        ],
    )


def get_centroid_closest_to_center(centroids: list[tuple[int, int]], center: int) -> int:
    # obtener centroide mas cercano al centro de la imagen
    closest = 0
    for i, centroid in enumerate(centroids):
        if abs(centroid[0] - center) < abs(centroids[closest][0] - center):
            closest = i
    return closest


def train_models(video_paths: list[str], read_from_disk: bool = False, size: int = 256) -> None:
    labels = []
    image_features = pd.DataFrame()
    for video_path in video_paths:
        print(f"Extracting features from {video_path}")
        # obtencion de los datos de entrenamiento
        l, frames = get_label_data(video_path)
        # obtener features de las imagenes
        img_features = get_image_features_from_video(video_path, frames, size)
        labels.append(l)
        image_features = pd.concat([image_features, img_features], ignore_index=True)
        # print(image_features)
    labels = np.concatenate(labels)
    # Entrenamiento de 3 modelos clasificadores (KNN, MLP, Random Forest)
    if read_from_disk:
        print("Reading models from disk...")
        # load the model from disk
        models = pickle.load(open("models.pkl", "rb"))
    else:
        print("Training models...")
        models = {}
        columns = ["Size", "State"]
        for i, col in enumerate(columns):
            print(f"Training models for {col}...")
            classifiers = [
                KNeighborsClassifier(n_neighbors=3),
                MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
                RandomForestClassifier(n_estimators=100),
                AdaBoostClassifier(),
                SVC(gamma="auto"),
            ]
            # entrenamiento de los modelos
            for classifier in classifiers:
                print(f"Training model {classifier.__str__()}...")
                classifier.fit(image_features, labels[:, i])
            models[col] = classifiers
        # save models to disk
        pickle.dump(models, open("models.pkl", "wb"))

    # Mostrar metricas de los modelos
    print("\nModel metrics:")
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\n{model_name}\n" + "-" * 30)
        for classifier in model:
            print(f"{classifier}")
            print(classifier.score(image_features, labels[:, i]))


def evaluate_video(video_path: str, size: int = 256) -> None:
    print(f"Evaluating video {video_path}...")
    models = pickle.load(open("models.pkl", "rb"))
    print(f"Models names: {models['Size']}")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (size, size))
        frame_features = get_image_features(frame)

        results = defaultdict(list)
        for model_name, model in models.items():
            for classifier in model:
                results[model_name].append(classifier.predict(frame_features))

        # dibujar una linea
        frame = cv2.line(frame, (size // 2, 0), (size // 2, size), (0, 0, 255), 2)
        # dibujar los resultados en la esquina superior izquierda
        for i, (model_name, result) in enumerate(results.items()):
            for j, r in enumerate(result):
                frame = cv2.putText(
                    frame,
                    f"{model_name} {r}",
                    (0, i * 40 + 20 + (j * 100)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    1,
                )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    size = 1024
    train_models(
        video_paths=[
            "dataset/lento_1.MOV",
            "dataset/lento_2.MOV",
            "dataset/lento_3.MOV",
            "dataset/med_1.MOV",
            "dataset/med_2.MOV",
            "dataset/med_3.MOV",
            "dataset/rapido_1.MOV",
        ],
        read_from_disk=False,
        size=size,
    )

    evaluate_video(
        video_path="dataset/lento_1.MOV",
        size=size,
    )
