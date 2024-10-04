from django.urls import path
from . import views
from . views import PostListView, PostDetailView, PostCreateView, PostUpdateView, PostDeleteView

urlpatterns=[
    #path('', views.home, name="bloghome"),
    path('about/', views.about, name="blogabout"),
    path('', PostListView.as_view(), name="bloghome"),
    path('post/new', PostCreateView.as_view(), name="blog-new"),
    path('post/<int:pk>/', PostDetailView.as_view(), name="blog-detail"),
    path('post/<int:pk>/update', PostUpdateView.as_view(), name="blog-update"),
    path('post/<int:pk>/delete', PostDeleteView.as_view(), name="blog-delete"),
    path('contact/', views.contact, name="contact"),
    path('test/', views.test, name="test"),
    path('test/result', views.result)
]