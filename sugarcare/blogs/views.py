from django.shortcuts import render
from .models import Post, Contact
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

#def home(request):
 #   context={
  #      'posts': Post.objects.all()
   # }
    #return render(request, 'blogs/home.html', context)


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
    data = pd.read_csv(r"C:\Users\espar\Downloads\diabetes.csv")
    
    # Split the dataset into features and target variable
    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']  # rbf kernel for non-linear relationships
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=0)
    grid.fit(X_train, Y_train)

    # Best model after hyperparameter tuning
    best_model = grid.best_estimator_

    # Evaluate model accuracy (optional, to confirm the model's performance)
    train_predictions = best_model.predict(X_test)
    train_accuracy = accuracy_score(Y_test, train_predictions)
    print(f"Model Accuracy: {train_accuracy * 100:.2f}%")  # Ensure it exceeds 90%

    # Get user input values
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    
    # Preprocess the user input using the same scaler
    user_input = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
    user_input_scaled = scaler.transform(user_input)

    # Make prediction using the trained model
    pred = best_model.predict(user_input_scaled)
    
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











