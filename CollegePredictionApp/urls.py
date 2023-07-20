from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
			path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
			path("UserLogin.html", views.UserLogin, name="UserLogin"),
			path("LoadDataset", views.LoadDataset, name="LoadDataset"),
			path("TrainML", views.TrainML, name="TrainML"),
			path("UserLoginAction.html", views.UserLoginAction, name="UserLoginAction"),
			path("Signup.html", views.Signup, name="Signup"),
			path("SignupAction", views.SignupAction, name="SignupAction"),
			path("PredictCollege.html", views.PredictCollege, name="PredictCollege"),
			path("PredictCollegeAction", views.PredictCollegeAction, name="PredictCollegeAction"),
				 
]