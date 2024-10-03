from django.shortcuts import render
from .models import Post, Contact
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages

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











