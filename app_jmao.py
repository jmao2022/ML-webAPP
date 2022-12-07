import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

def main():
    st.title("Binary Classification")    
    st.markdown("🍄左面：加载数据、选择模型、定义参数，然后点击run运行二分类计算🍄")
    st.markdown("🍄下面：显示输出信息🍄")
    st.markdown("🍄自动输出：文本信息output.txt，图outputTrain和outputTest🍄")
    st.sidebar.title("Binary Classification")
    st.sidebar.markdown("🍄 jmao 2022-12-06 🍄")

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv(Inputfile)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])        
        return data

    @st.cache(persist = True)
    def split(df):
        y = df[Inputlabel]
        x = df.drop([Inputlabel], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
        return x_train, x_test, y_train, y_test
    
    def plot_metricsTrain(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_train, y_train, display_labels = class_names)
            plt.savefig("outputTrain/Confusion_Matrix.png",bbox_inches='tight')
            plt.savefig("outputTrain/Confusion_Matrix.svg",bbox_inches='tight')
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_train, y_train)
            plt.savefig("outputTrain/ROC.png",bbox_inches='tight')
            plt.savefig("outputTrain/ROC.svg",bbox_inches='tight')
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_train, y_train)
            plt.savefig("outputTrain/Precision-Recall.png",bbox_inches='tight')
            plt.savefig("outputTrain/Precision-Recall.svg",bbox_inches='tight')
            st.pyplot()
    def plot_metricsTest(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            plt.savefig("outputTest/Confusion_Matrix.png",bbox_inches='tight')
            plt.savefig("outputTest/Confusion_Matrix.svg",bbox_inches='tight')
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            plt.savefig("outputTest/ROC.png",bbox_inches='tight')
            plt.savefig("outputTest/ROC.svg",bbox_inches='tight')
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            plt.savefig("outputTest/Precision-Recall.png",bbox_inches='tight')
            plt.savefig("outputTest/Precision-Recall.svg",bbox_inches='tight')
            st.pyplot()
    #判断保存图片的output文件夹存在
    if not os.path.exists("outputTrain"):
        os.mkdir("outputTrain")
    if not os.path.exists("outputTest"):
        os.mkdir("outputTest")        
    #隐藏警告信息
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #加载csv文件
    uploaded_file = st.sidebar.file_uploader("Choose a csv file 👇")
    if uploaded_file is not None:
        Inputfile = uploaded_file
        df = load_data()
    #定义label标签
    text_input = st.sidebar.text_input("Enter label name 👇")
    if text_input:
        Inputlabel = text_input
        x_train, x_test, y_train, y_test = split(df)
        class_names = ['type1', 'type0']
    #选择分类模型
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'auto')
    
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Run", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            model.fit(x_train, y_train)

            accuracy0 = model.score(x_train, y_train)
            y_pred0 = model.predict(x_train)
            st.write("Train Accuracy: ", accuracy0.round(2))
            st.write("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2))
            st.write("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2))
            plot_metricsTrain(metrics)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Test Accuracy: ", accuracy.round(2))
            st.write("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metricsTest(metrics)

            with open(f'outputSVM.txt','w') as f:
                print("Train Accuracy: ", accuracy0.round(2), file=f)
                print("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Test Accuracy: ", accuracy.round(2), file=f)
                print("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2), file=f)
                print("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2), file=f)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Run", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(x_train, y_train)

            accuracy0 = model.score(x_train, y_train)
            y_pred0 = model.predict(x_train)
            st.write("Train Accuracy: ", accuracy0.round(2))
            st.write("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2))
            st.write("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2))
            plot_metricsTrain(metrics)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Test Accuracy: ", accuracy.round(2))
            st.write("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metricsTest(metrics)

            with open(f'outputLR.txt','w') as f:
                print("Train Accuracy: ", accuracy0.round(2), file=f)
                print("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Test Accuracy: ", accuracy.round(2), file=f)
                print("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2), file=f)
                print("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2), file=f)         

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")        
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Run", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(x_train, y_train)

            accuracy0 = model.score(x_train, y_train)
            y_pred0 = model.predict(x_train)
            st.write("Train Accuracy: ", accuracy0.round(2))
            st.write("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2))
            st.write("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2))
            plot_metricsTrain(metrics)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Test Accuracy: ", accuracy.round(2))
            st.write("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metricsTest(metrics)

            with open(f'outputRF.txt','w') as f:
                print("Train Accuracy: ", accuracy0.round(2), file=f)
                print("Train Precision: ", precision_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Train Recall: ", recall_score(y_train, y_pred0, labels = class_names).round(2), file=f)
                print("Test Accuracy: ", accuracy.round(2), file=f)
                print("Test Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2), file=f)
                print("Test Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2), file=f)
                        
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
    
if __name__ == '__main__':
    main()
