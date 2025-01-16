from django.shortcuts import render
from .models import Post, Contact, Comments
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404


def about(request):
    return render(request, 'blogs/about.html', {'title':"About Page"})

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        content = request.POST.get('content')

        if len(name) < 2 or len(email) < 3 or len(phone) < 10 or len(content) < 4:
            messages.error(request, "Please fill the form correctly")
        else:
            contact = Contact(name=name, email=email, phone=phone, content=content)
            contact.save()
            messages.success(request, "Your message has been received")

    return render(request, 'blogs/contact.html', {'title': "Contact Page"})

def test(request):
    return render(request, 'blogs/test.html', {'title': "Testing Page"})

def explore(request):
    return render(request,'blogs/explore.html', {'title':"Explore More"} )

def meds(request):
    return render(request, 'blogs/meds.html', {'title':"Medications"})

def consult(request):
    return render(request, 'blogs/doctorpage.html', {'title':"Consult" })

def exercises(request):
    return render(request, 'blogs/exercise.html', {'title':"Yoga" })

#def result(request):
    #data= pd.read_csv(r"C:\Users\espar\Downloads\diabetes.csv")
    #X = data.drop("Outcome", axis=1) 
    #Y = data["Outcome"]
    #X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)
    #model= LogisticRegression(max_iter=500)
    #model.fit(X_train, Y_train)
    #val1= float(request.GET['n1'])
    #val2= float(request.GET['n2'])
    #val3= float(request.GET['n3'])
    #val4= float(request.GET['n4'])
    #val5= float(request.GET['n5'])
    #val6= float(request.GET['n6'])
    #val7= float(request.GET['n7'])
    #val8= float(request.GET['n8'])
    #pred= model.predict([[val1, val2,val3,val4,val5,val6,val7,val8]])
    #result2=""
    #if pred==[1]:
     #   result2="Positive"
    #else:
    #    result2="Negative"
    #return render(request, "blogs/test.html", {"result2": result2})

def result(request):
    # Load the dataset
    diabetes_dataset = pd.read_csv(r"C:\Users\espar\Downloads\diabetes.csv")
    
    # Define outlier thresholds for key columns
    outlier_ranges = {
        "Glucose": (50, 200),
        "BloodPressure": (40, 120),
        "SkinThickness": (10, 50),
        "Insulin": (15, 300),
        "BMI": (15, 50),
    }

    # Handle outliers by imputing with median
    for column, (low, high) in outlier_ranges.items():
        median_value = diabetes_dataset[column].median()
        diabetes_dataset.loc[diabetes_dataset[column] < low, column] = median_value
        diabetes_dataset.loc[diabetes_dataset[column] > high, column] = median_value

    # Split the dataset into features and target variable
    X = diabetes_dataset.drop("Outcome", axis=1)
    Y = diabetes_dataset["Outcome"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=2)
    X, Y = smote.fit_resample(X, Y)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Define SVM classifier with hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
    grid = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=5)
    grid.fit(X_train, Y_train)

    # Best classifier from grid search
    best_classifier = grid.best_estimator_

    # Predict on training data
    X_train_prediction = best_classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # Predict on testing data
    X_test_prediction = best_classifier.predict(X_test)
    testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # Print results
    print(f"Training Data Accuracy: {training_data_accuracy * 100:.2f}%")
    print(f"Testing Data Accuracy: {testing_data_accuracy * 100:.2f}%")

    # Get user input values
    val1 = float(request.POST['n1'])
    val2 = float(request.POST['n2'])
    val3 = float(request.POST['n3'])
    val4 = float(request.POST['n4'])
    val5 = float(request.POST['n5'])
    val6 = float(request.POST['n6'])
    val7 = float(request.POST['n7'])
    val8 = float(request.POST['n8'])


    # Preprocess the user input using the same scaler
    user_input = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
    user_input_scaled = scaler.transform(user_input)

    # Make prediction using the trained model
    pred = best_classifier.predict(user_input_scaled)

    # Interpret the result
    result2 = "Positive" if pred[0] == 1 else "Negative"

    # Render the result on the webpage
    return render(request, "blogs/test.html", {"result2": result2})



class PostListView(LoginRequiredMixin, ListView):
    model=Post
    template_name='blogs/home.html'
    context_object_name='posts'
    ordering=["-date_created"]


class PostDetailView(LoginRequiredMixin, DetailView):
    model=Post


class PostCreateView(LoginRequiredMixin, CreateView):
    model= Post
    fields=['title', 'content']

    def form_valid(self, form):
        form.instance.author= self.request.user
        return super().form_valid(form)
    
    
    

class PostUpdateView(LoginRequiredMixin,UserPassesTestMixin, UpdateView):
    model=Post
    fields=['title', 'content']

    def form_valid(self, form):
        form.instance.author= self.request.user
        return super().form_valid(form)
    
    
    def test_func(self):
        post=self.get_object()
        if self.request.user==post.author:
            return True
        return False
    
    

class PostDeleteView(DeleteView):
    model=Post
    success_url= '/'

    def test_func(self):
        post=self.get_object()
        if self.request.user==post.author:
            return True
        return False
    

class CommentCreateView(LoginRequiredMixin,CreateView):
    model=Comments
    fields=('comment',)
    template_name='blogs/comment.html'
    success_url= reverse_lazy('bloghome')


    def form_valid(self, form):
        form.instance.author = self.request.user
        form.instance.post = get_object_or_404(Post, pk=self.kwargs['pk'])
        return super().form_valid(form)











