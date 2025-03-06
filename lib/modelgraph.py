import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
import os
from math import ceil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score,precision_recall_fscore_support

##################### SCIKIT LEARN MODULE #########################################



def PlotConfusionMatrix(test_labels,predictions,label_names):
    """Picks a random image, plots it and labels it with a predicted and truth label.

    Args:
        test_labels: Array of labels for validation.
        predictions: Label predictions obtain from evaluaiting with any NN model.
        label_names: Specific names of the possible labels found in test_labels.

    Returns:
        A plot of a confusion matrix for the tested results from a model, displaying the
        labels and having gradients of colour to portrait the amount of these predcitions.
    """
    cm = confusion_matrix(y_true=test_labels,
                    y_pred=predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=label_names,
                                )
    disp.plot(cmap=plt.cm.Blues )
    plt.xticks(rotation=70)
    plt.gcf().set_size_inches(len(label_names), len(label_names))
    plt.show()


def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

##################### PLOTTING MODULE #########################################


def DisplayModelPredictions(model,dataset,class_list,represented_images=5,batch=1):
  """Displays a random image from the dataset and shows if the prediction label
     correlates with the image folder's label

  Args:
    model: Trained model with the dataset.
    dataset: dataset to be displayed and predicted in the representation.
    class_list : labels used in the dataset.
    represented_images: images to be displayed.

  """
  
  max_per_row = 5  # Max images per row

  if represented_images >=15: # Max value
     represented_images = 15

  dataset = dataset.skip(batch)

  for images, labels in dataset.take(1):  # Take first batch
      rows = ceil(represented_images / max_per_row)  # Compute required rows
      
      plt.figure(figsize=(15, rows * 3))  # Adjust height dynamically
      
      predicts = model.predict(images)
      
      for i in range(represented_images):
          plt.subplot(rows, max_per_row, i + 1)  # Dynamically create subplots
          
          label_index = labels[i].numpy().argmax()  # Convert one-hot to index
          predict_index = tf.argmax(predicts[i]).numpy()
          
          color = "green" if class_list[predict_index] == class_list[label_index] else "red"
          
          plt.imshow(images[i].numpy().astype("uint8"))
          plt.title(f"Class: {class_list[predict_index]}\nActual: {class_list[label_index]}", color=color)
          plt.axis("off") 
      
      plt.tight_layout()
      plt.show()

def PlotAccuracyAndLoss(history):
  """
  Plot the evolution of accuracy and loss of the model.
  Args:
  history: history data obtained from fitting a model.
  """ 
  val_loss = history.history['val_loss']
  val_accuracy = history.history['val_accuracy']
  loss = history.history['loss']
  accuracy = history.history['accuracy']

  epochs = range(len(history.history['loss']))

  plt.figure()
  plt.plot(epochs, loss, label='training loss')
  plt.plot(epochs, accuracy, label='training accuracy')
  plt.title('Training results')
  plt.xlabel('Epochs')
  plt.legend();

  plt.figure()
  plt.plot(epochs, val_loss, label='validation loss')
  plt.plot(epochs, val_accuracy, label='validation accuracy')
  plt.title('Validation results')
  plt.xlabel('Epochs')
  plt.legend();

def PlotAccuracyAndLossExtended(history_og,history_new,extendedvalue):
  """
  Plot the evolution of accuracy and loss of the model that has a extended training.
  Args:
  history: history data obtained from fitting a model from the first training part.
  history2: history data obtained from the second fitting.
  extendedvalue: mark that separates both history data trainings.
  """ 
  # Get original history measurements
  accuracy = history_og.history["accuracy"]
  loss = history_og.history["loss"]
 
  val_acc = history_og.history["val_accuracy"]
  val_loss = history_og.history["val_loss"]

  # Combine history data to be represented in one graph
  total_acc = accuracy + history_new.history["accuracy"]
  total_loss = loss + history_new.history["loss"]

  total_val_acc = val_acc + history_new.history["val_accuracy"]
  total_val_loss = val_loss + history_new.history["val_loss"]

  # Make plots
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label='Training Accuracy')
  plt.plot(total_val_acc, label='Validation Accuracy')
  plt.plot([extendedvalue-1, extendedvalue-1],
            plt.ylim(), label='Extenden training') 
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label='Training Loss')
  plt.plot(total_val_loss, label='Validation Loss')
  plt.plot([extendedvalue-1, extendedvalue-1],
            plt.ylim(), label='Extended training')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

def PlotAndCompareModels(history1,history2,model1_name="Model A"
                     ,model2_name="Model B",comparison="accuracy",
                     title="Model Comparison"):
  """
  Plot comparison of a certain history parameter between two models.

  Args:
  history1: history data obtained from fitting a model A.
  history2: history data obtained from fitting a model B.
  model1_name: name displayed on the graph for model A.
  model2_name:  name displayed on the graph for model B.
  comparison: value from history data to be compared.
  title: title to be displayed on the graph.

  """ 
  history1_value = history1.history[comparison]
  history2_value = history2.history[comparison]

  epochs = range(len(history1.history[comparison]))

  plt.figure()
  plt.plot(epochs, history1_value, label=model1_name)
  plt.plot(epochs, history2_value, label=model2_name)
  plt.title(title)
  plt.xlabel('Epochs')
  plt.legend();

def PlotAndCompareModelsFT(history1,history2,model1_name="Model A"
                     ,model2_name="Model B",
                     title="Model Comparison",finetune=0):
  """
  Plot comparison of a certain history parameter between two models.

  Args:
  history1: history data obtained from fitting a model A.
  history2: history data obtained from fitting a model B.
  model1_name: name displayed on the graph for model A.
  model2_name:  name displayed on the graph for model B.
  comparison: value from history data to be compared.
  title: title to be displayed on the graph.

  """ 

  epochs = range(len(history1))

  plt.figure()
  plt.plot(epochs, history1, label=model1_name)
  plt.plot(epochs, history2, label=model2_name)
  plt.plot([finetune-1, finetune-1],
            plt.ylim(), label='Fine tune training') 
  plt.title(title)
  plt.xlabel('Epochs')
  plt.legend();


def CompareModelScores(results1, results2):
  for key, value in results1.items():
    print(f"Baseline {key}: {value:.2f}, New {key}: {results2[key]:.2f}, Difference: {results2[key]-value:.2f}")



##################### OS MODULE #########################################
def ImagesInDir(path):
  """
  Prints image files included in directory path.
  Args:
  Path: main folder
    """ 
  for dirpath, dirnames, filenames in os.walk(path):
    print(f" {len(dirnames)} Directories and {len(filenames)} images in '{dirpath}'.")