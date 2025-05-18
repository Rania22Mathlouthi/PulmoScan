from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser # Assurez-vous que le modèle Account est importé

class RegistrationForm(UserCreationForm):
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('radiologue', 'Radiologue'),
    ]

    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True)

    class Meta:
        model = CustomUser  # Assurez-vous que votre modèle s'appelle Account
        fields = ['username', 'password1', 'password2', 'role']

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")

        # Validation des champs
        if not username:
            self.add_error('username', 'Le nom d\'utilisateur est requis.')
        if not password1 or not password2:
            self.add_error('password1', 'Les deux champs de mot de passe sont requis.')
        if password1 != password2:
            self.add_error('password2', 'Les mots de passe ne correspondent pas.')

        return cleaned_data
