from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import cv2
import pickle
import os


def load_images(path):
    images = []
    labels = []
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(os.path.basename(root))
            paths.append(os.path.join(root, file))

    classes = sorted(set(labels))
    return images, labels, classes, paths


def compute_descriptors(imgs):
    img_descriptors = []
    sift = cv2.SIFT_create()
    for i, img in enumerate(imgs):
        keypoints = sift.detect(img, None)
        _, desc = sift.compute(img, keypoints=keypoints, descriptors=None)
        img_descriptors.append(desc)

    return img_descriptors


def calculate_sample_weights(descriptors, labels):
    """
    This is used for weighted k-means implementation.
    """
    class_weights = dict()
    for label in labels:
        if label in class_weights.keys():
            class_weights[label] += 1
        else:
            class_weights[label] = 1
    for label in class_weights.keys():
        class_weights[label] = len(labels) / class_weights[label]

    sample_weights = []
    for i in range(len(labels)):
        if descriptors[i] is None:
            continue
        sample_weights.extend([class_weights[labels[i]]] * len(descriptors[i]))

    return sample_weights


def find_dictionary(descriptors, k=50, save_model=False):
    """
    This function fits k-means model using sift descriptors
    """
    model = KMeans(n_clusters=k, random_state=5)
    # flatten the list to construct a dataset for clustering
    data = [desc for img_descriptors in descriptors if img_descriptors is not None for desc in img_descriptors]
    data = np.array(data)
    model = model.fit(data)

    if save_model:
        with open("kmeans_model_k" + str(k) + ".pkl", "wb") as model_file:
            pickle.dump(model, model_file)

    return model


def load_dictionary_from_file(path):
    """
    K-means takes too long to converge, this function is used to load previously learned models.
    """
    with open(path, "rb") as model_file:
        model = pickle.load(model_file)

    return model


def quantize(kmeans_model, descriptors):
    """
    Using kmeans model, descriptors can be assigned to closest clusters to form a feature histogram
    """
    k = kmeans_model.get_params()["n_clusters"]
    img_histogram = np.zeros((len(descriptors), k))
    for i, descriptor_ in enumerate(descriptors):
        if descriptor_ is None:
            continue
        assigned_clusters = kmeans_model.predict(descriptor_)
        for cno in assigned_clusters:
            img_histogram[i][cno] += 1

    normed_img_histogram = img_histogram / np.linalg.norm(img_histogram)

    return normed_img_histogram


def compute_kernel(data1, data2):
    """
    Pre-computation for chi-square kernel
    """
    computed_data = chi2_kernel(X=data1, Y=data2)
    return computed_data


def train_classifier(data, labels, kernels):
    """
    This is where SVM classifier is trained and paramter search is done.
    This function returns the best model and all other parameter sets for experiments.
    """

    precomputed_data = compute_kernel(data1=data, data2=data)

    all_search_params = []
    best_result = 0.0
    best_params = None
    best_model = None

    for kernel in kernels:
        base_svc = SVC()
        parameters = {"class_weight": ["balanced", None], "kernel": [kernel],
                      "C": [1, 10, 50, 100], "random_state": [5]}
        svm_model = GridSearchCV(base_svc, param_grid=parameters, cv=5, n_jobs=-1, refit=True)
        if kernel == "precomputed":
            svm_model = svm_model.fit(precomputed_data, labels)
        else:
            svm_model = svm_model.fit(data, labels)

        all_search_params.extend(svm_model.cv_results_["params"])

        if best_result < svm_model.best_score_:
            best_result = svm_model.best_score_
            best_params = svm_model.best_params_
            best_model = svm_model

    all_search_params.remove(best_params)

    return best_model, best_params, all_search_params


def run_all_experiments(best_model, best_params, parameter_list, test_data, train_data,
                        test_labels, train_labels, classes):

    print("\n---------Test Results---------")
    precomputed_train_data = compute_kernel(data1=train_data, data2=train_data)
    precomputed_test_data = compute_kernel(data1=test_data, data2=train_data)

    for param_set in parameter_list:
        print(param_set)
        svm_model = SVC(**param_set)
        if param_set["kernel"] == "precomputed":
            svm_model.fit(precomputed_train_data, train_labels)
            test(svm_model, precomputed_test_data, test_labels, classes)
        else:
            svm_model.fit(train_data, train_labels)
            test(svm_model, test_data, test_labels, classes)

    print("\nResults of the Best Model")
    print(best_params)
    best_res_misclassification, predicted_labels = test(best_model, precomputed_test_data, test_labels, classes)

    return best_res_misclassification, predicted_labels


def test(model, test_data, true_labels, classes):

    predicted_labels = model.predict(test_data)
    test_classes = sorted(set(true_labels))

    average_f1_score = f1_score(true_labels, predicted_labels, average="macro")
    class_f1_score = f1_score(true_labels, predicted_labels, average=None, labels=test_classes)
    confusion_matrix_ = confusion_matrix(true_labels, predicted_labels, labels=classes)

    print("Macro Av. F1-Score: " + str(average_f1_score))
    print("Class F1-Score:")
    print(*[label + ":" + "{:.2f}".format(score) for label, score in zip(test_classes, class_f1_score)], sep=", ")
    print("Confusion Matrix")
    print(confusion_matrix_)
    errors = [idx for idx in range(len(true_labels)) if true_labels[idx] != predicted_labels[idx]]

    return errors, predicted_labels


def show_error_example(test_img_paths, error_indices, predicted_labels, save_img=False):
    """
    Shows 20x20 thumbnails of example wrong predictions
    """

    images = dict()
    for idx in error_indices:
        img = cv2.imread(test_img_paths[idx])
        img = cv2.resize(img, (20, 20))
        if predicted_labels[idx] in images.keys():
            images[predicted_labels[idx]].append(img)
        else:
            images[predicted_labels[idx]] = [img]

    category_concat = []
    label_order = []
    for label in images.keys():
        combinedtwo = np.full((20, 40, 3), 256)
        hconcatted = np.concatenate(images[label][:2], axis=1)
        combinedtwo[:, :hconcatted.shape[1], :] = hconcatted
        category_concat.append(combinedtwo)
        label_order.append(label)

    all_concat = np.concatenate(category_concat, axis=0)

    if save_img:
        cv2.imwrite("misclass_examples.jpeg", all_concat)
        print("Image saved to: misclass_examples.jpeg")
        print("Predicted Labels (top to bottom): " + " ".join(label_order))
    else:
        print("Misclassification results image is not saved!")


if __name__ == '__main__':
    # Parameters
    dictionary_size = 500

    # Train
    train_images, train_labels, train_classes, _ = load_images("Caltech20/training")
    train_descriptors = compute_descriptors(train_images)
    print("Images Descriptors Computed!")
    # comment below to load a learned clustering model on the next lines
    kmeans_model_ = find_dictionary(train_descriptors, k=dictionary_size, save_model=True)
    # uncomment below to load a learned clustering model
    # kmeans_model_ = load_dictionary_from_file("kmeans_model_k500.pkl")
    print("Clustering Completed! k=" + str(kmeans_model_.get_params()["n_clusters"]))
    train_histogram = quantize(kmeans_model_, train_descriptors)
    print("Train Feature Histograms Computed!")
    classifier, best_params_, param_list = train_classifier(train_histogram, train_labels,
                                                            kernels=["precomputed", "linear"])
    print("Classifier Training Completed!")

    # Test
    test_images, test_labels, _, test_paths = load_images("Caltech20/testing")
    test_descriptors = compute_descriptors(test_images)
    test_histogram = quantize(kmeans_model_, test_descriptors)
    print("Test Feature Histograms Computed!")
    error_indices_, predicted_labels_ = run_all_experiments(classifier, best_params_, param_list,
                        test_histogram, train_histogram, test_labels, train_labels, train_classes)
    print("Experiments Completed!")
    # example misclassified image thumbnail will be saved to current directory
    show_error_example(test_paths, error_indices_, predicted_labels_, save_img=True)
