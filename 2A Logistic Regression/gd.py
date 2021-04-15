import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def normalization(data):
    '''
    Function to normalise the features to 0 to 1  
    '''
    minval=np.amin(data,axis=0)     # axis=0 returns an array containing the smallest element for each column
    maxval=np.amax(data,axis=0)
    # Iterate over each column (feature) and compute the normalized value of each value in feature column
    for i in range(0,len(minval)-1):
        data[:,i]=(data[:,i]-minval[i])/(maxval[i]-minval[i])
    return data


def train_test_split(data):
    '''
    Function to randomly split the data into training and testing; 70:30
    '''
    np.random.shuffle(data)    # Shuffle datapoints randomly
    # Split training and testing data in 70:30 ratio
    div = (int) (0.7* data.shape[0])
    training_set=data[0:div,:]      
    testing_set=data[div:,:]
    return (training_set,testing_set)

def metrics1(product, x, y_org):
    '''
    Function to calculate the Accuracy and Loss 
    '''
    y = np.floor(product + 0.5) # Predicated Classses

    no_misclassified = np.sum(abs(y - y_org))
    accuracy = 1 - (no_misclassified / x.shape[0])
    loss = np.sum(-(np.multiply(y_org, np.log(product)) + np.multiply(1-y_org, np.log(1-product))))
    

    return [accuracy, loss]

def metrics2(product, x, y_org): 
    '''
    Function to calculate the other metrics
    '''   
    y = np.floor(product + 0.5) # Predicated Classses
    TP =  (np.sum(np.logical_and(y_org, y)))
    FN = (np.sum(np.logical_and(y_org, np.logical_not(y))))
    precision = (TP)/(np.sum(y)) 
    recall = (TP)/(TP + FN )
    fscore = 2 * ((precision * recall)/(precision + recall))

    return [precision, recall, fscore]

def sigmoid(w, x):
    '''
    Function to apply sigmoid transformation 
    '''
    sigma = 1/(1+np.exp(-x.dot(w)))
    return sigma


def gd(x, y_org, lr, num_iter ):
    '''
    Function to apply Gradient descent on training data, and find weights 
    '''
    w = np.ones((4,1))
    xaxis = []
    error_to_plot = []
    loss_to_plot = []
    
    pdt = sigmoid(w, x)
    diff = pdt - y_org

    for i in range(0, num_iter):
        grad = np.dot(x.T, diff)
        #print(grad)
        w = w - lr*(grad)
        pdt = sigmoid(w, x)

        diff = pdt - y_org

        if(i%50 == 0):
            xaxis.append(i)
            [iter_error, loss] = metrics1(pdt, x, y_org)
            #loss = np.sum(-(np.multiply(y_org, np.log(pdt)) + np.multiply(1-y_org, np.log(1-pdt))))
            #print("Iteration ", i, "; Accuracy = ", iter_error, "; Loss = ", loss)
            loss_to_plot.append(loss)
            error_to_plot.append(iter_error)

    plt.figure(1)
    plt.subplot(2,1,1) 
    plt.plot(xaxis, error_to_plot)
    plt.title("Learning rate = " + str(lr))
    plt.ylabel("Accuracy")

    plt.subplot(2,1,2)
    plt.plot(xaxis, loss_to_plot)
    plt.xlabel("No. of Iterations")
    plt.ylabel("Loss")


    #plt.figure(2) 
    #plt.plot(xaxis, loss_to_plot,  label = "Learning rate = " + str(lr))
    #plt.show()
    #plt.close()
    return w

def process(data, learning_rate, num_iter):
    '''
    Function to make the data ready for Gradient Descent 
    '''
    # Training data part
    (training_data,testing_data) = train_test_split(data)
    x_train=training_data[:,0:4]
    y_train=training_data[:,4]
    #ones=np.ones((x.shape[0],1))
    #x=np.append(ones,x,axis=1)
    y_train=np.reshape(y_train,(x_train.shape[0],1))
    x_test=testing_data[:,0:4]
    #x_test=np.append(ones,x_test,axis=1)
    y_test_org=testing_data[:,4]
    y_test_org=np.reshape(y_test_org,(x_test.shape[0],1))


    weight = gd(x_train, y_train, learning_rate, num_iter)
    print("Weights = ", weight)
    # Train data part
    pdt_train = sigmoid(weight, x_train)
    [accuracy_train, loss_train] = metrics1(pdt_train, x_train, y_train)
    [precision_train, recall_train, fscore_train] = metrics2(pdt_train, x_train, y_train)
    train_metrics = [accuracy_train, loss_train, precision_train, recall_train, fscore_train]
    print("Train Accuracy = ", accuracy_train)

    # Test data part
    pdt_test = sigmoid(weight, x_test)
    [accuracy_test, loss_test] = metrics1(pdt_test, x_test, y_test_org)
    [precision_test, recall_test, fscore_test] = metrics2(pdt_test, x_test, y_test_org)
    test_metrics = [accuracy_test, loss_test, precision_test, recall_test, fscore_test]
    print("Test Accuracy = ", accuracy_test)
    
    return train_metrics, test_metrics





def main():
    data=pd.read_csv("./data/dataset_LR.csv").to_numpy()
    data = normalization(data)

    lr = 0.0001           # Learning Rate
    num_iter = 10000  # Number Of Iterations

    train_accuracy_list=[] 
    train_loss_list = []
    train_precision_list = []
    train_recall_list = []
    train_fscore_list = []

    test_accuracy_list=[] 
    test_loss_list = []
    test_precision_list = []
    test_recall_list = []
    test_fscore_list = []

    start = time.time()
    for i in range(0,10):
        print("Split ",i)
        (train_metrics, test_metrics) = process(data, lr, num_iter)

        train_accuracy_list.append(train_metrics[0])
        train_loss_list.append(train_metrics[1])
        train_precision_list.append(train_metrics[2])
        train_recall_list.append(train_metrics[3])
        train_fscore_list.append(train_metrics[4])

        test_accuracy_list.append(test_metrics[0])
        test_loss_list.append(test_metrics[1])
        test_precision_list.append(test_metrics[2])
        test_recall_list.append(test_metrics[3])
        test_fscore_list.append(test_metrics[4])
        


    end = time.time()

    plt.show()
    plt.close()

    train_final = {'Accuracy':train_accuracy_list,'Loss':train_loss_list, 'Precision':train_precision_list, 'Recall': train_recall_list,'F-score':train_fscore_list}
    train_final_df = pd.DataFrame(train_final)
    train_final_df = train_final_df.rename_axis('Train Data', axis=1)
    print(train_final_df)
    print(train_final_df.describe().iloc[1]) # To Display Mean

    print("\n\n")

    test_final = {'Accuracy':test_accuracy_list,'Loss':test_loss_list, 'Precision':test_precision_list, 'Recall': test_recall_list,'F-score':test_fscore_list}
    test_final_df = pd.DataFrame(test_final)
    test_final_df = test_final_df.rename_axis('Test Data', axis=1)
    print(test_final_df)
    print(test_final_df.describe().iloc[1]) # To Display Mean


    print("Time = ",((end-start)))

if __name__ == "__main__":
    main()
    